classdef ONNXTensorLabeler < handle
    % A class to propagate ONNX tensor labels through an ONNX graph
    
    % Copyright 2021-2025 The MathWorks, Inc.

    properties(Access=protected)
        % The full graph
        GraphProtoManager nnet.internal.cnn.onnx.GraphProtoManager
        
        % The operator translator objects
        OperatorTranslators nnet.internal.cnn.onnx.OperatorTranslator
        
        % A map from ONNX tensor names to ONNX format strings (labels)
        TensorLabels % containers.Map, char-->string
    end
    methods
        function this = ONNXTensorLabeler(graphProtoManager, operatorTranslators, userInputLabels, userOutputLabels)
            this.GraphProtoManager      = graphProtoManager;
            this.OperatorTranslators    = operatorTranslators;
            this.TensorLabels           = containers.Map('KeyType', 'char', 'ValueType', 'any');
            % Set seed labels and propagate internally first
            setLabelsFromSeedOperators(this);
            propagateAllTensorLabels(this);
            % Apply user-provided labels and check for conflicts
            applyUserLabels(this, userInputLabels, userOutputLabels);
            % Propagate again with user labels added
            propagateAllTensorLabels(this);            
        end
         
        function str = tensorLabel(this, tensorName)
            if isKey(this.TensorLabels, char(tensorName))
                str = this.TensorLabels(char(tensorName));
            else
                str = "";
            end
        end
        
        function inputTensorFormats = nodeInputFormats(this, nodeNum)
            inputTensorFormats = string.empty;
            inputNames = nodeInputNames(this.GraphProtoManager, nodeNum);
            % Performance note: Keep this loop. It is 6x faster than using
            % arrayfun with an anonymous function, and this method is
            % called many times for large models.
            for i=numel(inputNames):-1:1
                inputTensorFormats(i) = tensorLabel(this, inputNames(i));
            end
        end
        
        function outputTensorFormats = nodeOutputFormats(this, nodeNum)
            outputTensorFormats = string.empty;
            outputNames = nodeOutputNames(this.GraphProtoManager, nodeNum);
            % Performance note: Keep this loop. It is 6x faster than using
            % arrayfun with an anonymous function, and this method is
            % called many times for large models.
            for i=numel(outputNames):-1:1
                outputTensorFormats(i) = tensorLabel(this, outputNames(i));
            end
        end
    end
    
    methods(Access=protected)
        function applyUserLabels(this, userInputLabels, userOutputLabels)
            % Set the labels of the graph's inputs and outputs, if they
            % were provided by the user. 
            externalInputNames = this.GraphProtoManager.ExternalInputNames;
            externalOutputNames = this.GraphProtoManager.OutputNames;
            assert(isempty(userInputLabels) || numel(userInputLabels)==numel(externalInputNames), ...
                message('nnet_cnn_onnx:onnx:IncorrectNumInputLabels', numel(userInputLabels), numel(externalInputNames)));
            assert(isempty(userOutputLabels) || numel(userOutputLabels)==numel(externalOutputNames),...
                message('nnet_cnn_onnx:onnx:IncorrectNumOutputLabels', numel(userOutputLabels), numel(externalOutputNames)));
            userInputLabels = string(userInputLabels);
            userOutputLabels = string(userOutputLabels);
            % Assign input labels
            for i = 1:numel(userInputLabels)
                if userInputLabels(i) ~= ""
                    % The user provided a label for this input.
                    % Check its numDims.
                    tensorName = externalInputNames{i};
                    internalLabel = tensorLabel(this, tensorName);
                    if nnet.internal.cnn.onnx.util.validateIONDataFormatsNumDims(...
                        onnxInputSizeCell(this.GraphProtoManager, tensorName), true, i, userInputLabels(i), internalLabel)
                        setTensorLabel(this, tensorName, userInputLabels(i));
                    end
                end
            end
            % Assign output labels
            for i = 1:numel(userOutputLabels)
                if userOutputLabels(i) ~= ""
                    % Use user provided a label for this output.
                    % Check its numDims.
                    tensorName = externalOutputNames{i};
                    internalLabel = tensorLabel(this, tensorName);
                    if nnet.internal.cnn.onnx.util.validateIONDataFormatsNumDims(...
                        onnxOutputSizeCell(this.GraphProtoManager, tensorName), false, i, userOutputLabels(i), internalLabel)
                        setTensorLabel(this, tensorName, userOutputLabels(i));
                    end
                end
            end
        end

        function propagateAllTensorLabels(this)
            % Label as many tensors as possible in the graph. 
            anyLabelsChanged = true;
            while anyLabelsChanged
                anyLabelsChangedFwd = propagateLabelsOneDirection(this, true);
                anyLabelsChangedBwd = propagateLabelsOneDirection(this, false);
                anyLabelsChanged = anyLabelsChangedFwd || anyLabelsChangedBwd;
            end
        end
        
        function setTensorLabel(this, tensorName, value)
            this.TensorLabels(char(tensorName)) = string(value);
        end
                
        function anyLabelChanged = setNodeInputFormats(this, nodeNum, currentInputTensorFormats, newInputTensorFormats, canOverwrite)
            % Sets input formats to new values if present.
            inputNames = nodeInputNames(this.GraphProtoManager, nodeNum);
            anyLabelChanged = false;
            for i=1:numel(inputNames)
                if newInputTensorFormats(i)~="" && (currentInputTensorFormats(i)=="" || canOverwrite)
                    setTensorLabel(this, inputNames(i), newInputTensorFormats(i));
                    if newInputTensorFormats(i) ~= currentInputTensorFormats(i)
                        anyLabelChanged = true;
                    end
                end
            end
        end
                
        function anyLabelChanged = setNodeOutputFormats(this, nodeNum, currentOutputTensorFormats, newOutputTensorFormats, canOverwrite)
            % Sets output formats to new values if present.
            outputNames = nodeOutputNames(this.GraphProtoManager, nodeNum);
            anyLabelChanged = false;
            for i=1:numel(outputNames)
                if newOutputTensorFormats(i)~="" && (currentOutputTensorFormats(i)=="" || canOverwrite)
                    setTensorLabel(this, outputNames(i), newOutputTensorFormats(i));
                    if newOutputTensorFormats(i) ~= currentOutputTensorFormats(i)
                        anyLabelChanged = true;
                    end
                end
            end
        end

        function setLabelsFromSeedOperators(this)
            % Set labels from seed operators only
            nodeIndices = 1:this.GraphProtoManager.NumNodes;          	% topological order
            for nodeNum = nodeIndices
                opTranslator = this.OperatorTranslators(nodeNum);
                if opTranslator.IsSeedOperator
                    currentInputFormats = nodeInputFormats(this, nodeNum);
                    currentOutputFormats = nodeOutputFormats(this, nodeNum);
                    [updatedInputFormats, updatedOutputFormats] = propagateTensorFormats(opTranslator, "forward", currentInputFormats, currentOutputFormats);
                    setNodeInputFormats(this, nodeNum, currentInputFormats, updatedInputFormats, opTranslator.CanOverwriteTensorLabels);
                    setNodeOutputFormats(this, nodeNum, currentOutputFormats, updatedOutputFormats, opTranslator.CanOverwriteTensorLabels);
                end
            end
        end
        
        function anyLabelsChanged = propagateLabelsOneDirection(this, goForward)
            % Propagate tensor labels through all nodes in one direction.
            anyLabelsChanged = false;
            if goForward
                nodeIndices = 1:this.GraphProtoManager.NumNodes;          	% topological order
                direction = "forward";
            else
                nodeIndices = this.GraphProtoManager.NumNodes:-1:1;      	% reverse topological order
                direction = "backward";
            end
            for nodeNum = nodeIndices
                opTranslator = this.OperatorTranslators(nodeNum);
                currentInputFormats = nodeInputFormats(this, nodeNum);
                currentOutputFormats = nodeOutputFormats(this, nodeNum);
                if direction=="forward"
                    fromFormats = currentInputFormats;
                else
                    fromFormats = currentOutputFormats;
                end
                if ~iAnySingletonFormats(fromFormats) || opTranslator.CanPropagateSingletonFormats
                [updatedInputFormats, updatedOutputFormats] = propagateTensorFormats(opTranslator, direction, currentInputFormats, currentOutputFormats);
                    labelChangedInput = setNodeInputFormats(this, nodeNum, currentInputFormats, updatedInputFormats, opTranslator.CanOverwriteTensorLabels);
                    labelChangedOutput = setNodeOutputFormats(this, nodeNum, currentOutputFormats, updatedOutputFormats, opTranslator.CanOverwriteTensorLabels);
                    anyLabelsChanged = anyLabelsChanged || labelChangedInput || labelChangedOutput;
                end
                end
            end
        end
    end

function tf = iAnySingletonFormats(formatStrings)
tf = any(contains(formatStrings, '1'));
end
