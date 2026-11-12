classdef ModelTranslationIntoLayers < handle

    % Copyright 2021-2026 The MathWorks, Inc.

    properties (SetAccess=protected)
        % The result of the translation: a layer graph
        LayerGraph nnet.cnn.LayerGraph

        % Issues encountered during translation
        TranslationIssues nnet.internal.cnn.onnx.NodeTranslationIssue

        % Settings from the user
        CreateCustomLayers logical
        GenerateCoderFiles logical
        OutputLayerType
        TargetNetwork
        ImageInputSize
        Classes
        PackageName
        OrigPackageName
        FoldConstants

        % The format string for each input tensor. A string array. Inputs
        % with unknown formats will have the value ""
        InputFormats string

        % The format string for each output tensor. A string array. Outputs
        % with unknown formats will have the value ""
        OutputFormats string

        % Names of all output tensors for this model. A cell array. These
        % are present in the underlying GraphProtoManager
        ModelOutputNames cell
        
        % number of output layers that's needed in this model. 
        NumOutputLayers int64

        % output names for dlnetwork
        DlnetworkOutputNames cell

        % Flag for whether the importer expects Forward or Reverse ONNX
        % ordering for all U-dimension inputs
        IsInputForwardONNX logical

        % How to handle grouping of nodes when creating custom layers.
        CustomLayerContent string
    end

    properties(Access=private)
        % A cell array of layer arrays for individual nodes
        LayerCell = {};

        % A cell array of issue arrays for individual nodes
        IssueCell = {};

        % A layer array containing all layers to be added
        % (non-sequentially)
        LayerArray = [];  % A column vector

        % A table of all the connections to be added
        LayerConnectionTable = table('Size',[0 2],'VariableTypes',{'string','string'},'VariableNames',{'Source','Destination'});
    end

    properties(Dependent)
        OpsetVersion;
    end

    properties (Access=protected)
        % Internal Settings
        MinSupportedOpset = 6;
        MaxSupportedOpset = 20;
        MaxNameLength = 20;

        % The ModelProto
        ModelProto nnet.internal.cnn.onnx.ModelProto

        % The model's top-level GraphProto
        GraphProtoManager nnet.internal.cnn.onnx.GraphProtoManager

        % An object containing ONNX tensor labels and the names of fully-supported nodes
        OnnxTensorLabeler nnet.internal.cnn.onnx.ONNXTensorLabeler

        % An object responsible for creating custom layers
        CustomLayerManager nnet.internal.cnn.onnx.CustomLayerManager

        % An array of operator translators, in the same order as nodes in
        % the graph
        OperatorTranslators nnet.internal.cnn.onnx.OperatorTranslator

        % A logical vector in 1-1 correspondence with the nodes of the
        % graph
        IsNodeFullySupported logical

        % A map from onnx tensor name to DLT output connection string.
        TensorCreatorMap % containers.Map, char-->char

        % A map from DLT layer input port name to ONNX tensor name
        LayerInputTensorNameMap % containers.Map

        % A flag to check if the model needs a verifyBatchSizeLayer
        NeedVerifyBatchSize = false;
    end

    methods(Static)
        %% Translating between DLT and ONNX format strings
        function dltTensorFormat = onnxLabelToDLTLabel(onnxTensorLabel)
            switch onnxTensorLabel
                case ""
                    dltTensorFormat = "";
                case "BC"                             % feature batch: CB
                    dltTensorFormat = "CB";
                case {"BCSS" "BSSC" "CSS" "SSC"}      % 2d image batch: SSCB
                    dltTensorFormat = "SSCB";
                case {"BCSSS" "BSSSC" "CSSS" "SSSC"}  % 3d image batch: SSSCB
                    dltTensorFormat = "SSSCB";
                case "1BC"                            % vector sequence batch's last time step: CB1=CB
                    dltTensorFormat = "CB";
                case {"T1BC" "TBC" "BTC" "BCT"}       % vector sequence batch output from an ONNX LSTM operator: CBT
                    dltTensorFormat = "CBT";
                case "TBCSS"                          % 2d image sequence batch: SSCBT
                    dltTensorFormat = "SSCBT";
                case "TBCSSS"                         % 3d image sequence batch: SSSCBT
                    dltTensorFormat = "SSSCBT";
            end
        end

        function permToONNX = permDLTToONNX(targetOnnxFormat)
            % The permutation vector to permute a tensor from DLT to ONNX
            % when the target ONNX format is targetOnnxFormat. These should
            % work in a context in which the DLT array and ONNX tensors are
            % both N-D MATLAB arrays.

            % It's important that doing the permutation would result in a
            % MATLAB array with the same number of dimensions as the ONNX
            % format string (or fewer). The B==1 cases use this fact.
            switch targetOnnxFormat
                case "BC"       % from CB
                    permToONNX = [2 1];
                case "BCSS"   	% from SSCB
                    permToONNX = [4 3 1 2];
                case "BSSC"     % from SSCB
                    permToONNX = [4 1 2 3];
                case "CSS"      % from SSCB, B==1
                    permToONNX = [3 1 2];
                case "SSC"      % from SSCB, B==1
                    permToONNX = [1 2 3];
                case "BCSSS"    % from SSSCB
                    permToONNX = [5 4 1 2 3];
                case "BSSSC"    % from SSSCB
                    permToONNX = [5 1 2 3 4];
                case "CSSS"     % from SSSCB, B==1
                    permToONNX = [4 1 2 3];
                case "SSSC"     % from SSSCB, B==1
                    permToONNX = [1 2 3 4];
                case "BCT"      % from CBT
                    permToONNX = [2 1 3];
                case "TBC"      % from CBT
                    permToONNX = [3 2 1];
                case "BTC"      % from CBT
                    permToONNX = [2 3 1];
                case "1BC"      % from CB
                    permToONNX = [3 2 1];
                case "T1BC"     % from CBT
                    permToONNX = [3 4 2 1];
                case "TBCSS"    % from SSCBT
                    permToONNX = [5 4 3 1 2];
                case "TBCSSS"   % from SSSCBT
                    permToONNX = [6 5 4 1 2 3];
                otherwise
                    assert(false, "Unknown target ONNX tensor format");
            end
        end

        function permToDLT = permONNXToDLT(onnxFormat)
            % The permutation vector to permute a tensor from ONNX to DLT
            % when the ONNX format is onnxFormat. These should work in a
            % context in which the DLT array and ONNX tensors are both N-D
            % MATLAB arrays
            switch onnxFormat
                case "BC" % to CB
                    permToDLT = [2 1];
                case "BCSS" % to SSCB
                    permToDLT = [3 4 2 1];
                case "BSSC" % to SSCB
                    permToDLT = [2 3 4 1];
                case "CSS" % to SSCB, B=1
                    permToDLT = [2 3 1];
                case "SSC" % to SSCB, B=1
                    permToDLT = [1 2 3];
                case "BCSSS" % to SSSCB
                    permToDLT = [3 4 5 2 1];
                case "BSSSC" % to SSSCB
                    permToDLT = [2 3 4 5 1];
                case "CSSS" % to SSSCB, B=1
                    permToDLT = [2 3 4 1];
                case "SSSC" % to SSSCB, B=1
                    permToDLT = [1 2 3 4];
                case "BCT" % to CBT
                    permToDLT = [2 1 3];
                case "TBC" % to CBT
                    permToDLT = [3 2 1];
                case "BTC" % to CBT
                    permToDLT = [3 1 2];
                case "1BC" % to CB
                    permToDLT = [3 2 1];
                case "T1BC" % to CBT
                    permToDLT = [4 3 1 2];
                case "TBCSS" % to SSCBT
                    permToDLT = [4 5 3 2 1];
                case "TBCSSS" % to SSSCBT
                    permToDLT = [4 5 6 3 2 1];
                otherwise
                    assert(false, "Unknown ONNX tensor format '" + onnxFormat + "'");
            end
        end
    end

    methods
        function this = ModelTranslationIntoLayers(modelProto, outputLayerType, ...
                userInputLabels, userOutputLabels, targetNetwork, createCustomLayers, ...
                imageInputSize, classes, packageName, origPackageName, foldConsts, isInputForwardONNX,...
                CustomLayerContent, GenerateCoderFiles)
            this.CreateCustomLayers = createCustomLayers;
            this.OutputLayerType    = outputLayerType;
            this.TargetNetwork      = targetNetwork;
            this.ImageInputSize     = imageInputSize;
            this.Classes            = classes;
            this.PackageName        = packageName;
            this.OrigPackageName    = origPackageName;
            this.LayerGraph         = layerGraph();
            this.TranslationIssues  = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            this.ModelProto         = modelProto;
            this.TensorCreatorMap   = containers.Map('KeyType','char', 'ValueType','char');
            this.LayerInputTensorNameMap = containers.Map('KeyType','char', 'ValueType','any');
            this.FoldConstants      = foldConsts;
            this.IsInputForwardONNX = isInputForwardONNX;
            this.CustomLayerContent = CustomLayerContent;
            this.GenerateCoderFiles = GenerateCoderFiles;

            iValidateOpsetVersion(this.OpsetVersion, this.MaxSupportedOpset, this.MinSupportedOpset);
            % Clear the persistent variable used to generate new names
            import nnet.internal.cnn.onnx.fcn.*;
            clear uniqueName;
            clear uniqueLayerName;
            clear uniqueCoderName;
            % Preprocess the graph
            this.GraphProtoManager = nnet.internal.cnn.onnx.GraphProtoManager(this.ModelProto.graph, this.OpsetVersion);
            makeGraphNamesDLTCompatible(this.GraphProtoManager, this.MaxNameLength);
            foldConstants(this.GraphProtoManager, this.FoldConstants);
            % Translate the supported nodes
            createOperatorTranslators(this);
            this.OnnxTensorLabeler = nnet.internal.cnn.onnx.ONNXTensorLabeler(this.GraphProtoManager, this.OperatorTranslators, userInputLabels, userOutputLabels);
            translateAllSupportedNodes(this);
            % Generate custom layers
            if this.CreateCustomLayers
                clear uniqueName;
                FullySupportedNodeNames = this.GraphProtoManager.NodeNames(this.IsNodeFullySupported);
                this.CustomLayerManager = nnet.internal.cnn.onnx.CustomLayerManager(this.GraphProtoManager, ...
                                          this.OpsetVersion, this.OnnxTensorLabeler, FullySupportedNodeNames, ...
                                          this.PackageName, this.OrigPackageName, this.CustomLayerContent, this.GenerateCoderFiles, false);
                this.TranslationIssues = [this.TranslationIssues, getCustomLayerTranslationIssues(this.CustomLayerManager)];
            end
            % Before processing input & output layers, change output type
            % to dlnetwork if there are any coupled custom layers
            if isequal(this.TargetNetwork, 'dagnetwork') && ~isempty(this.CustomLayerManager) && this.CustomLayerManager.AnyCoupledCustomLayers
                warning(message('nnet_cnn_onnx:onnx:CantAssembleDAGNetwork'));
                this.TargetNetwork = "dlnetwork";
            end
            % Add and connect layers
            addInputLayers(this);
            addHiddenLayers(this);
            addOutputLayers(this);
            addRemainingConnections(this);
            assembleLayerGraph(this);
            findInputOutputTensorFormats(this);
            if checkMinLengthRequired(this)
                autoSetMinLength(this);
            end
            addFixedBatchSizeLayers(this);
            % Generate coder custom layers            
            if this.CreateCustomLayers && this.GenerateCoderFiles
                % Clear the persistent variable used to generate new names
                clear uniqueName;
                clear uniqueLayerName;
                clear uniqueCoderName;
                nnet.internal.cnn.onnx.CustomLayerManager(this.GraphProtoManager, ...
                                          this.OpsetVersion, this.OnnxTensorLabeler, FullySupportedNodeNames, ...
                                          this.PackageName, this.OrigPackageName, this.CustomLayerContent, this.GenerateCoderFiles, true);
            end
        end

        % Getters
        function ver = get.OpsetVersion(this)
            if isempty(this.ModelProto.opset_import)
                ver = 1;
            else
                ver = max(double([this.ModelProto.opset_import.version]));
            end
        end

        function outputNames = get.ModelOutputNames(this)
            outputNames = this.GraphProtoManager.OutputNames;
        end
    end

    methods(Access=protected)
        function layerDigraph = createLayerDigraph(this)
            % Create a digraph using this.LayerGraph's connections
            % table
            srcStrings = string(this.LayerGraph.Connections.Source);
            destStrings = string(this.LayerGraph.Connections.Destination);

            % Remove '/in' and '/out' for layers with multiple inputs or
            % outputs.
            for i=1:numel(srcStrings)
                src = split(srcStrings(i), '/');
                srcStrings(i) = src(1);
            end
            for i=1:numel(destStrings)
                dest = split(destStrings(i), '/');
                destStrings(i) = dest(1);
            end

            % Construct digraph
            layerDigraph = digraph(srcStrings, destStrings);
        end

        function tf = graphHasMultipleComponents(this)
            layerDigraph = createLayerDigraph(this);
            ccbins = conncomp(layerDigraph, 'Type', 'weak');
            tf = max(ccbins) > 1;
        end

        function createOperatorTranslators(this)
            for nodeNum = 1:this.GraphProtoManager.NumNodes
                node = nodesFromIdx(this.GraphProtoManager, nodeNum);
                this.OperatorTranslators(nodeNum) = nnet.internal.cnn.onnx.OperatorTranslator.create(node, this.GraphProtoManager, this.CreateCustomLayers, this.TargetNetwork);
            end
        end

        function translateAllSupportedNodes(this)
            % Sets this.LayerCell, this.IssueCell, and
            % this.IsNodeFullySupported. Each element of LayerCell is an
            % array of layers for one node. Each element of IssueCell is an
            % array of issues for one node.
            for nodeNum = 1:this.GraphProtoManager.NumNodes
                [layerArray, issueArray] = translateIntoLayer(this.OperatorTranslators(nodeNum), ...
                    nodeInputFormats(this.OnnxTensorLabeler, nodeNum), ...
                    nodeOutputFormats(this.OnnxTensorLabeler, nodeNum));
                if this.OperatorTranslators(nodeNum).NeedVerifyBatchSize
                    this.NeedVerifyBatchSize = true;
                end
                this.LayerCell{nodeNum} = layerArray;
                this.IssueCell{nodeNum} = issueArray;
            end
            if isequal(this.CustomLayerContent, "entireNetwork")
                % Mark all nodes as if they are unsupported for import into
                % built-in layers
                this.IsNodeFullySupported = cellfun(@(layerArray)false, this.LayerCell);
            else
                this.IsNodeFullySupported = cellfun(@(layerArray)~isempty(layerArray), this.LayerCell);
            end
        end

        function addInputLayers(this)
            % Add input layers to this.LayerArray
            numInputTensors = numel(this.GraphProtoManager.ExternalInputNames);
            for inputNum = 1:numInputTensors
                inputTensorName = this.GraphProtoManager.ExternalInputNames{inputNum};
                % Create an input layer
                if inputNum==1
                    % Apply ImageInputSize if present to the first input
                    userImageInputSize = this.ImageInputSize;
                else
                    userImageInputSize = [];
                end
                Layer = createInputLayer(this, inputTensorName, userImageInputSize);
                % Add the layer to the layer array
                this.LayerArray = [this.LayerArray; Layer(:)];
                %  Update the tensor creator map
                if isa(Layer, 'nnet.onnx.layer.CustomInputLayerMultiOutput')
                    % This layer includes a second output for NumDims
                    this.TensorCreatorMap(Layer.OutputNames{1}) = char(Layer.Name + "/" + Layer.OutputNames{1});
                    this.TensorCreatorMap(Layer.OutputNames{2}) = char(Layer.Name + "/" + Layer.OutputNames{2});
                else
                    this.TensorCreatorMap(char(inputTensorName)) = char(Layer.Name);
                end
                % Add an issue for placeholder layers
                if isa(Layer, 'nnet.cnn.layer.PlaceholderLayer')
                    ONNXInputSizeCell = onnxInputSizeCell(this.GraphProtoManager, inputTensorName);
                    ONNXInputSizeCell = iReplaceEmptyCells(ONNXInputSizeCell, "?");
                    ONNXDimString = "()";
                    if ~isempty(ONNXInputSizeCell)
                        ONNXDimString = "(" + join(string(ONNXInputSizeCell), ', ') + ")";
                    end
                    ONNXNode = struct;
                    ONNXNode.name = Layer.Name;
                    ONNXNode.op_type = 'input';
                    if inputNum==1 && isImageTensor(this, inputTensorName)
                        issue = nnet.internal.cnn.onnx.NodeTranslationError(ONNXNode, message('nnet_cnn_onnx:onnx:ImageInputSizeUnknown', inputNum, Layer.Name));
                    else
                        issue = nnet.internal.cnn.onnx.NodeTranslationError(ONNXNode, message('nnet_cnn_onnx:onnx:InputLayerConstructorFailed', inputNum, Layer.Name, ONNXDimString));
                    end
                    this.TranslationIssues(end+1) = issue;
                end
            end
        end

        function tf = isImageTensor(this, inputTensorName)
            % Return true if the input tensor is an image tensor
            % representing either a single image or an image batch
            imageFormats = [...
                "BCSS" "BSSC" "CSS" "SSC" ...       % 2d image batch: SSCB
                "BCSSS" "BSSSC" "CSSS" "SSSC"...    % 3d image batch: SSSCB
                ];
            tf  = ismember(tensorLabel(this.OnnxTensorLabeler, inputTensorName), imageFormats);
        end

        function tf = isSingleImageTensor(this, inputTensorName)
            % Return true if the input tensor is an image tensor
            % representing a single image
            imageFormats = [...
                "CSS" "SSC" ...     % A single 2d image: SSCB
                "CSSS" "SSSC"...    % A single 3d image: SSSCB
                ];
            tf  = ismember(tensorLabel(this.OnnxTensorLabeler, inputTensorName), imageFormats);
        end

        function addFixedBatchSizeLayers(this)
            % Insert a VerifyBatchSizeLayer layer into the layerGraph
            % right after the DLT input layers if an ONNX input tensor has 
            % a fixed batch size, but is being represented by 
            % a DLT input layer that has a variable batch size when the model
            % is a DAGNetwork or this.NeedVerifyBatchSize is set to true
            if ~isequal(this.TargetNetwork, 'dlnetwork') || this.NeedVerifyBatchSize
                layerNames = {this.LayerGraph.Layers.Name};
                [~, inputLayerIdx] = ismember(this.LayerGraph.InputNames, layerNames);
                inputLayers = this.LayerGraph.Layers(inputLayerIdx);
                variableBatchSizeDims = arrayfun(@iDLTVariableBatchSizeDim, inputLayers);  % Number of the batch dimension for each DLT input layer, or NaN if no batch dimension.
                fixedBatchSizes = onnxFixedBatchSizes(this);                            % The fixed batch sizes for each ONNX input tensor, or NaN if not fixed.
                for i=1:numel(inputLayers)
                    if ~isnan(fixedBatchSizes(i)) && ~isnan(variableBatchSizeDims(i))
                        % Insert a VerifyBatchSizeLayer after this input layer
                        verifierLayerName = string(inputLayers(i).Name) + "_BatchSizeVerifier";
                        verifierLayer = nnet.onnx.layer.VerifyBatchSizeLayer(verifierLayerName, fixedBatchSizes(i), variableBatchSizeDims(i), inputLayers(i).Name);
                        layerPair = [inputLayers(i), verifierLayer];
                        this.LayerGraph = replaceLayer(this.LayerGraph, layerNames{i}, layerPair);
                    end
                end
            end
        end

        function fixedBatchSizes = onnxFixedBatchSizes(this)
            % Returns an array of batch sizes for fixed-batch-size ONNX
            % input tensors, or NaNs for inputs without fixed batch size
            gpm = this.GraphProtoManager;
            otl = this.OnnxTensorLabeler;
            numInputTensors = numel(gpm.ExternalInputNames);
            fixedBatchSizes = NaN(1,numInputTensors);
            for i=1:numInputTensors
                inputTensorName = gpm.ExternalInputNames{i};
                if isSingleImageTensor(this, inputTensorName)
                    % This input is a single image
                    fixedBatchSizes(i) = 1;
                else
                    % The input is a minibatch. Find the batch size
                    fmt = tensorLabel(otl, inputTensorName);
                    tensorSizeCell = gpm.onnxInputSizeCell(inputTensorName);
                    locB = strfind(fmt,'B');
                    if ~isempty(locB) && locB<=numel(tensorSizeCell) && isnumeric(tensorSizeCell{locB}) && ~isempty(tensorSizeCell{locB})
                        fixedBatchSizes(i) = tensorSizeCell{locB};
                    end
                end
            end
        end

        function addOutputLayers(this)
            % Add output layers to this.LayerArray.
            % Add real output layers to dagnetworks, and custom layers to
            % dlnetworks
            dlNetworkOutputNames = cell(1,this.GraphProtoManager.NumOutputs);
            numOutputLayers = 0;
            seenOutputTensors = dictionary(string.empty, int32.empty);
            for outputNum = 1:this.GraphProtoManager.NumOutputs
                outputTensorName = this.GraphProtoManager.OutputNames{outputNum};
                % Check if this is a duplicate
                isDuplicate = isKey(seenOutputTensors, outputTensorName);

                if isDuplicate
                    duplicateCount = seenOutputTensors(outputTensorName) + 1;
                    seenOutputTensors(outputTensorName) = duplicateCount;
                else
                    seenOutputTensors(outputTensorName) = 0;
                    duplicateCount = 0;
                end
                if outputNum==1
                    % Apply the user's classes and output layer type to the
                    % first output layer
                    userClasses = this.Classes;
                    userOutputLayerType = this.OutputLayerType;
                else
                    userClasses = [];
                    userOutputLayerType = [];
                end
                % Create the layer
                if this.TargetNetwork=="dagnetwork"
                    Layer = createDAGOutputLayer(this, outputTensorName, userClasses, userOutputLayerType);
                else % dlnetwork
                    Layer = createOutputPermuteLayer(this, outputTensorName, isDuplicate, duplicateCount);
                end
                if ~isempty(Layer)
                    % Add the layer to the layer array
                    this.LayerArray = [this.LayerArray; Layer(:)];
                    % Add the input port and tensor name to the input map
                    this.LayerInputTensorNameMap(Layer.Name + "/in") = outputTensorName;
                    % Set the dlnetwork output name this layer's name
                    dlNetworkOutputNames{outputNum} = Layer.Name;
                    numOutputLayers = numOutputLayers + 1;
                elseif isKey(this.TensorCreatorMap, outputTensorName)
                        % Set the dlnetwork output name to be the creator
                        % of this output tensor
                        source = this.TensorCreatorMap(outputTensorName);
                        dlNetworkOutputNames{outputNum} = source;
                else
                    % Set the dlnetwork output name to be this output
                    % tensor
                    dlNetworkOutputNames{outputNum} = outputTensorName;
                end
                % Add an issue for placeholders
                if isa(Layer, 'nnet.cnn.layer.PlaceholderLayer')
                    dummyNode.name = "";
                    dummyNode.op_type = "";
                    issue = nnet.internal.cnn.onnx.NodeTranslationError(dummyNode, message('nnet_cnn_onnx:onnx:OutputLayerConstructorFailed', outputNum, outputTensorName));
                    this.TranslationIssues(end+1) = issue;
                end
            end
            this.NumOutputLayers = numOutputLayers;
            this.DlnetworkOutputNames = dlNetworkOutputNames;
        end

        function findInputOutputTensorFormats(this)
            for inputNum = 1:numel(this.GraphProtoManager.ExternalInputNames)
                inputTensorName = this.GraphProtoManager.ExternalInputNames{inputNum};
                % Add the input format to the object property
                onnxFormat = tensorLabel(this.OnnxTensorLabeler, inputTensorName);
                this.InputFormats(inputNum) = nnet.internal.cnn.onnx.ModelTranslationIntoLayers.onnxLabelToDLTLabel(onnxFormat);
            end
            for outputNum = 1:this.GraphProtoManager.NumOutputs
                outputTensorName = this.GraphProtoManager.OutputNames{outputNum};
                % Add the output format to the object property
                onnxFormat = tensorLabel(this.OnnxTensorLabeler, outputTensorName);
                this.OutputFormats(outputNum) = nnet.internal.cnn.onnx.ModelTranslationIntoLayers.onnxLabelToDLTLabel(onnxFormat);
            end
        end

        function addHiddenLayers(this)
            % Add hidden layers to this.LayerArray. Add connections to
            % this.LayerConnectionTable
            hiddenLayersCell = {};
            for nodeNum = 1:numel(this.GraphProtoManager.GraphProto.node)
                thisNode = this.GraphProtoManager.GraphProto.node(nodeNum);
                layerArray = [];
                layerIssues = [];
                if this.IsNodeFullySupported(nodeNum)
                    layerArray = this.LayerCell{nodeNum};
                    layerIssues = this.IssueCell{nodeNum};
                    % The layerInputTensorNames are the ONNX tensor names
                    % for the node inputs that end up being DLT layer
                    % inputs. Those are just the inputs that are not
                    % initializers:
                    isInitializer = isTensorInitializer(this.GraphProtoManager, thisNode.input);
                    isEmptyInput = cellfun(@isempty, thisNode.input);
                    layerInputTensorNames = thisNode.input(~isInitializer & ~isEmptyInput);
                    inputConnStrings = iGetInputConnectionStrings(layerArray(1));
                    isEmptyOutput = cellfun(@isempty, thisNode.output);
                    outputTensorNames = thisNode.output(~isEmptyOutput);
                    outputConnStrings = iGetOutputConnectionStrings(~isEmptyOutput, layerArray(end));
                    % Record any layerArray-internal connections
                    if numel(layerArray) > 1
                        for i=1:numel(layerArray)-1
                            source = layerArray(i).Name + "/" + layer.OutputNames{1};
                            dest = layerArray(i+1).Name + "/" + layer.InputNames{1};
                            this.LayerConnectionTable{end+1,:} = {source, dest};
                        end
                    end
                elseif this.CreateCustomLayers
                    % This node is not supported, and will be put inside an
                    % auto-generated custom layer.
                    LayerName = nodeCustomLayerName(this.CustomLayerManager, thisNode.name);
                    if isNodeLastInSubgraph(this.CustomLayerManager, thisNode.name, LayerName)
                        % The custom layer has not yet been added to the
                        % layergraph, and this is the last node in the
                        % subgraph. Add the layer now.
                        layerConstructor = str2func(string(this.PackageName) + "." + string(LayerName));
                        layerArray = layerConstructor(LayerName);
                        onnxParam = onnxParamsForLayerName(this.CustomLayerManager, LayerName);

                        learnableFieldNames = fieldnames(onnxParam.Learnables);
                        for i = 1:numel(learnableFieldNames)
                            layerArray.(learnableFieldNames{i}) = onnxParam.Learnables.(learnableFieldNames{i});
                        end
                        nonLearnableFieldNames = fieldnames(onnxParam.Nonlearnables);
                        for i = 1:numel(nonLearnableFieldNames)
                            layerArray.Vars.(nonLearnableFieldNames{i}) = onnxParam.Nonlearnables.(nonLearnableFieldNames{i});
                        end
                        stateFieldNames = fieldnames(onnxParam.State);
                        for i = 1:numel(stateFieldNames)
                            layerArray.(stateFieldNames{i}) = extractdata(onnxParam.State.(stateFieldNames{i}));
                        end
                        % onnxParam.NumDimensions returns [] when it's
                        % empty. Check whether it's empty before trying to
                        % find the name. 
                        if ~isempty(onnxParam.NumDimensions)
                            numDimsNames = fieldnames(onnxParam.NumDimensions);
                            for i = 1:numel(numDimsNames)
                                layerArray.NumDims.(numDimsNames{i}) = onnxParam.NumDimensions.(numDimsNames{i});
                            end
                        end
                        layerInputTensorNames = [...
                            customLayerInputTensorNames(this.CustomLayerManager, layerArray.Name),...
                            customLayerInputNumDimsNames(this.CustomLayerManager, layerArray.Name)];
                        inputConnStrings = iMakeInputConnectionStrings(layerArray.Name, layerArray.NumInputs);
                        outputTensorNames = [...
                            customLayerOutputTensorNames(this.CustomLayerManager, layerArray.Name),...
                            customLayerOutputNumDimsNames(this.CustomLayerManager, layerArray.Name)];
                        outputConnStrings = iMakeCustomOutputConnectionStrings(outputTensorNames, layerArray.Name);
                    end
                else
                    % This node is not supported, and CreateCustomLayers is
                    % false, so it will be imported into a PlaceholderLayer
                    LayerName = thisNode.name;
                    [layerArray, layerIssues] = translateUnsupportedONNXLayers(this, thisNode, LayerName);
                    isInitializer = isTensorInitializer(this.GraphProtoManager, thisNode.input);
                    isEmptyInput = cellfun(@isempty, thisNode.input);
                    layerInputTensorNames = thisNode.input(~isInitializer & ~isEmptyInput);
                    inputConnStrings = iMakeInputConnectionStrings(LayerName, numel(layerInputTensorNames));
                    outputTensorNames = iGetNonEmptyNames(thisNode.output);
                    outputConnStrings = iMakeOutputConnectionStrings(thisNode, layerArray(end).Name);
                end
                % Add the layerArray for this node to hiddenLayersCell
                hiddenLayersCell = [hiddenLayersCell, arrayfun(@(x)x,layerArray,'UniformOutput',false)];
                % Update tensor maps and issues
                if ~isempty(layerArray)
                    % Update the tensor maps
                    for i=1:numel(inputConnStrings)
                        this.LayerInputTensorNameMap(inputConnStrings(i)) = layerInputTensorNames{i};
                    end
                    % Some output tensors may not connect to anything. So
                    % loop over outputConnStrings instead of
                    % outputTensorNames here
                    for i=1:numel(outputConnStrings)
                        this.TensorCreatorMap(char(outputTensorNames{i})) = char(outputConnStrings(i));
                    end
                    % Add this node's issues to the list
                    this.TranslationIssues = [this.TranslationIssues layerIssues];
                end
            end
            % Finally, add hiddenLayersCell to this.LayerArray
            this.LayerArray = [this.LayerArray; [hiddenLayersCell{:}]'];   % Append to column vector
        end

        function addRemainingConnections(this)
            % Add remaining connections to this.LayerConnectionTable
            for i=1:numel(this.LayerArray)
                layer = this.LayerArray(i);
                % Connect this layer's inputs
                for inputNum = 1:numel(layer.InputNames)
                    inputConnString = layer.Name + "/" + layer.InputNames{inputNum};
                    if isKey(this.LayerInputTensorNameMap, inputConnString)
                        % Find the name of the tensor coming into this
                        % input port
                        inputTensorName = this.LayerInputTensorNameMap(inputConnString);
                        if isKey(this.TensorCreatorMap, inputTensorName)
                            % Find the DLT layer output port that creates
                            % this tensor
                            source = this.TensorCreatorMap(inputTensorName);
                            % Record a connection from the source output
                            % port to the destination input port
                            dest = inputConnString;
                            this.LayerConnectionTable{end+1,:} = {source, dest};
                        end
                    end
                end
            end
        end

        function assembleLayerGraph(this)
            %             % Create the LayerGraph using the fast loadobj technique
            %             s = [];
            %             s.Layers = this.LayerArray;
            %             s.Connections = this.LayerConnectionTable;
            %             this.LayerGraph = nnet.cnn.LayerGraph.loadobj(s);

            numInputLayers = numel(this.GraphProtoManager.ExternalInputNames);
            numOutputLayers = this.NumOutputLayers;
            numHiddenLayers = numel(this.LayerArray) - numInputLayers - numOutputLayers;

            % (1) Add input layers individually
            lg = this.LayerGraph;
            for i=1:numInputLayers
                lg = addLayers(lg, this.LayerArray(i));
            end
            % (2) Add all hidden layers at once (producing some spurious
            % connections)
            if numHiddenLayers>0
                lg = addLayers(lg, this.LayerArray(numInputLayers+1 : numInputLayers+numHiddenLayers));
            end
            % (3) Add output layers individually
            for i=1:numOutputLayers
                lg = addLayers(lg, this.LayerArray(numInputLayers+numHiddenLayers+i));
            end
            % (3) Delete connections not in the desired connection table,
            % and add missing connections
            desiredConns = [string(this.LayerConnectionTable.Source), string(this.LayerConnectionTable.Destination)];
            currentSource = string(lg.Connections.Source);
            % Add true output port names to currentSource strings
            for i=1:numel(currentSource)
                if ~contains(currentSource(i),'/')
                    sourceLayerIndex = numInputLayers + i;  % Because the connections are 1-1 with the hidden layers at this point.
                    outName = this.LayerArray(sourceLayerIndex).OutputNames{1};
                    currentSource(i) = currentSource(i) + "/" + outName;
                end
            end
            currentDest = string(lg.Connections.Destination);
            % Add "/in" port name to Destinations that don't have a port name
            for i=1:numel(currentDest)
                if ~contains(currentDest(i),'/')
                    currentDest(i) = currentDest(i) + "/in";
                end
            end
            % Remove and add connections as needed
            currentConns = [currentSource, currentDest];
            connsToRemove = currentConns;
            if ~isempty(currentConns) && ~isempty(desiredConns)
                connsToRemove = setdiff(currentConns, desiredConns, 'rows', 'stable');
            end
            for i=1:size(connsToRemove,1)
                lg = disconnectLayers(lg, connsToRemove(i,1), connsToRemove(i,2));
            end

            connsToAdd = desiredConns;
            if ~isempty(desiredConns) && ~isempty(currentConns)
                connsToAdd = setdiff(desiredConns, currentConns, 'rows', 'stable');
            end
            for i=1:size(connsToAdd,1)
                lg = connectLayers(lg, connsToAdd(i,1), connsToAdd(i,2));
            end
            this.LayerGraph = lg;

            % Finally, add a ConnectComponents layer if needed.
            if graphHasMultipleComponents(this)
                addConnectComponentsLayer(this);
            end
        end

        function addConnectComponentsLayer(this)
            % DLT does not allow a single network to have multiple
            % disconnected components. So insert a layer just before
            % the output layers to connect the components.
            connCompLayerName = "ConnectComponentsLayer";
            connCompLayer = nnet.onnx.layer.ConnectComponentsLayer(connCompLayerName, this.GraphProtoManager.NumOutputs);
            this.LayerGraph = addLayers(this.LayerGraph, connCompLayer);
            isOutputLayer = getOutputLayersIdx(this);
            outputLayerNames = {this.LayerGraph.Layers(isOutputLayer).Name};
            % Connect the connComp layer:
            [~,idx] = ismember(outputLayerNames, this.LayerGraph.Connections.Destination);
            sources = this.LayerGraph.Connections.Source(idx);
            for i=1:numel(sources)
                source = sources{i};
                source = iWorkaroundSingleOutputSourceIssue(this.LayerGraph.Layers, source);
                this.LayerGraph = disconnectLayers(this.LayerGraph, source, outputLayerNames{i});
                this.LayerGraph = connectLayers(this.LayerGraph, source, connCompLayerName + "/in" + i);
                this.LayerGraph = connectLayers(this.LayerGraph, connCompLayerName + "/out" + i, outputLayerNames{i});
            end
        end

        function outputLayersIdx = getOutputLayersIdx(this)
            % Return logical indices of the output layers in the network
            isCustomOutput = arrayfun(@(L)isa(L, 'nnet.onnx.layer.CustomOutputLayer'), this.LayerGraph.Layers);
            isToolboxOutput = arrayfun(@(L)ismember(L.Name, this.LayerGraph.OutputNames), this.LayerGraph.Layers);
            outputLayersIdx = isCustomOutput | isToolboxOutput;
        end

        function [NNTLayer, issues] = translateUnsupportedONNXLayers(this, node, LayerName)
            issues = nnet.internal.cnn.onnx.NodeTranslationError(node,...
                message('nnet_cnn_onnx:onnx:UnsupportedOperator', node.op_type));
            if isempty(node.input)
                WeightList = [];
                numInputs = 1;
            else
                inputWithoutInitializer = ~isTensorInitializer(this.GraphProtoManager, node.input);
                isEmptyInput = cellfun(@isempty, node.input);
                WeightList = ~inputWithoutInitializer & ~isEmptyInput;
                numInputs = max(1, sum(inputWithoutInitializer & ~isEmptyInput));
            end
            numOutputs = max(1, numel(node.output));
            NNTLayer =  nnet.onnx.layer.PlaceholderLayer(LayerName, node, numInputs, numOutputs);
            NNTLayer.Weights = [];
            % import weights
            WeightNameList = node.input(WeightList);
            for i = 1:numel(WeightNameList)
                WeightName = WeightNameList{i};
                if isempty(initializerSize(this.GraphProtoManager, WeightName)) || isempty(initializerRawData(this.GraphProtoManager, WeightName))
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationWarning(node,...
                        message('nnet_cnn_onnx:onnx:EmptyInitializer'))]; %#ok<*AGROW>
                end
                WeightDim = initializerSize(this.GraphProtoManager, WeightName);
                WeightValue = single(initializerRawData(this.GraphProtoManager, WeightName));
                nDim = numel(WeightDim);
                if nDim > 1
                    WeightValue = reshape(WeightValue, fliplr(WeightDim));
                    WeightValue = permute(WeightValue, nDim:-1:1);
                end
                NNTLayer.Weights(i).name = WeightName;
                NNTLayer.Weights(i).value = WeightValue;
            end
        end

        function createdInputLayer = createInputLayer(this, inputTensorName, userImageInputSize)
            % Extract ONNX tensor metadata
            ONNXInputSizeCell = onnxInputSizeCell(this.GraphProtoManager, inputTensorName);
            DLTTensorName = nnet.internal.cnn.onnx.makeNNTName(inputTensorName);
            DLTLayerName      = DLTTensorName;
            inputTensorLabel  = tensorLabel(this.OnnxTensorLabeler, inputTensorName);
            % Initialize outputs
            createdInputLayer = [];

            switch inputTensorLabel
                case "BC"
                    % Feature vector batch: CB format in MATLAB
                    C = ONNXInputSizeCell{end};
                    if isnumeric(C) && ~isempty(C)
                        DLTInputShape = double(C);
                        createdInputLayer = featureInputLayer(DLTInputShape, 'Name', DLTLayerName);
                    end
                case {"BCSS" "BSSC" "CSS" "SSC"}
                    % 2D image batch: SSCB format in MATLAB
                    DLTInputShape = iGetDLTImageInputSize(ONNXInputSizeCell, inputTensorLabel, userImageInputSize);
                    if ~isempty(DLTInputShape)
                        createdInputLayer = imageInputLayer(DLTInputShape, 'Name', DLTLayerName, 'Normalization', 'none');
                    end
                case {"BCSSS" "BSSSC" "CSSS" "SSSC"}
                    % 3D image batch: SSSCB format in MATLAB
                    DLTInputShape = iGetDLTImageInputSize(ONNXInputSizeCell, inputTensorLabel, userImageInputSize);
                    if ~isempty(DLTInputShape)
                        createdInputLayer = image3dInputLayer(DLTInputShape, 'Name', DLTLayerName, 'Normalization', 'none');
                    end
                case {"TBC","BTC","BCT"}
                    % Sequence input with various orderings
                    if isequal(inputTensorLabel,"TBC")
                        T = ONNXInputSizeCell{1};
                        C = ONNXInputSizeCell{end};
                    elseif isequal(inputTensorLabel,"BTC")
                        % 1D networks from Keras to ONNX
                        T = ONNXInputSizeCell{2};
                        C = ONNXInputSizeCell{end};
                    else  % BCT
                        % 1D networks from PyTorch to ONNX
                        T = ONNXInputSizeCell{end};
                        C = ONNXInputSizeCell{2};
                    end

                    if isnumeric(C) && ~isempty(C)
                        DLTInputShape = double(C);
                        if ~isnumeric(T) || isempty(T)
                            createdInputLayer = sequenceInputLayer(DLTInputShape, 'Name', DLTLayerName);
                        else
                            createdInputLayer = sequenceInputLayer(DLTInputShape, 'Name', DLTLayerName,'MinLength',T);
                        end
                    end
                case {"TBCSS" "TBCSSS"}
                    % Image sequence batch
                    DLTInputShape = iGetDLTImageInputSize(ONNXInputSizeCell, inputTensorLabel, userImageInputSize);
                    if ~isempty(DLTInputShape)
                        createdInputLayer = sequenceInputLayer(DLTInputShape, 'Name', DLTLayerName);
                    end
            end
            if isempty(createdInputLayer)
                if this.TargetNetwork == "dagnetwork"
                    % DAGNetwork: use placeholder layer
                    createdInputLayer = nnet.onnx.layer.PlaceholderInputLayer(DLTLayerName);
                else
                    % dlnetwork: create custom or generic input layers
                    if inputTensorLabel~=""
                        dltTensorFormat         = nnet.internal.cnn.onnx.ModelTranslationIntoLayers.onnxLabelToDLTLabel(inputTensorLabel);
                        permToDLT               = nnet.internal.cnn.onnx.ModelTranslationIntoLayers.permONNXToDLT(inputTensorLabel);
                        % Identify spatial/channel dimensions that must have fixed sizes
                        idxSCU = ismember(char(dltTensorFormat), 'SC');
                        inputSize = ONNXInputSizeCell;
                        % Prepare input size array, replacing invalid entries with NaN
                        inputSize(cellfun(@iIsInvalidSize, inputSize)) = {NaN};
                        inputSize = cell2mat(inputSize);
                        % Check if we can use generic inputLayer
                        % Requires: dimensions match format length AND spatial/channel dims are fixed
                        if numel(permToDLT) == strlength(dltTensorFormat) && ~any(isnan(inputSize(idxSCU)))
                            % Use generic inputLayer
                            createdInputLayer = inputLayer(inputSize, dltTensorFormat, "Name", DLTLayerName);
                        else
                            % Use CustomInputLayer for dimension mismatch
                            onnxInfoInDLTOrdering   = iReplaceEmptyCells(ONNXInputSizeCell(permToDLT), "?");
                            textlines = [getString(message('nnet_cnn_onnx:onnx:LabeledInputDescriptionL1')),...
                                "    '" + dltTensorFormat + "'",...
                                getString(message('nnet_cnn_onnx:onnx:LabeledInputDescriptionL2')),...
                                "    (" + join(string(onnxInfoInDLTOrdering), ', ') + ")"];
                            inputInformation = join(textlines, newline);
                            createdInputLayer       = nnet.onnx.layer.CustomInputLayer(DLTLayerName, inputInformation, dltTensorFormat);
                        end
                    else
                        % SUBSECTION 2B: Unlabeled input (format unknown)
                        hasAllSizes = ~any(cellfun(@iIsInvalidSize, ONNXInputSizeCell));
                        if this.IsInputForwardONNX && hasAllSizes
                            % when input has no empty size
                            inputSize = ONNXInputSizeCell;
                            if numel(ONNXInputSizeCell)<2
                                % Pad to 2 dimensions
                                len = numel(ONNXInputSizeCell);
                                inputSize = [repmat({1},1,2-len), inputSize];
                                inputSize          = cell2mat(inputSize);
                                createdInputLayer  = inputLayer(inputSize, 'UU', "Name", DLTLayerName);
                            else
                                % Multi-dimensional case: need permutation from forward to reverse ONNX
                                % Remove trailing singleton dimensions for
                                % inputLayer requirement
                                inputSize = cell2mat(inputSize);
                                while numel(inputSize) > 2 && inputSize(end) == 1
                                    inputSize(end) = [];
                                end
                                numDims = numel(inputSize);
                                UString = string(repmat('U', 1, numDims));
                                inputLayers       = inputLayer(inputSize, UString, "Name", "in");
                                % Permute dimensions in reverse order
                                inputLayers(2)    = permuteLayer(numel(ONNXInputSizeCell):-1:1,"Name","out");
                                createdInputLayer = networkLayer(inputLayers, "Name", DLTLayerName);
                            end
                        else
                            % Unknown format and/or unknown sizes: use CustomInputLayerMultiOutput
                            % This layer outputs both the tensor and its number of dimensions

                            % Determine dimension ordering for display
                            if this.IsInputForwardONNX
                                onnxInfoInDLTOrdering   = iReplaceEmptyCells(ONNXInputSizeCell, "?");
                            else
                                onnxInfoInDLTOrdering   = fliplr(iReplaceEmptyCells(ONNXInputSizeCell, "?"));
                            end
                            if this.IsInputForwardONNX
                                inputInfoOrderL1 = getString(message('nnet_cnn_onnx:onnx:UnlabeledInputDescriptionL1NewAPI'));
                            else
                                inputInfoOrderL1 = getString(message('nnet_cnn_onnx:onnx:UnlabeledInputDescriptionL1'));
                            end

                            % Pad to 2 dimensions if numDim is smaller than 2
                            if numel(ONNXInputSizeCell)<2
                                len = numel(ONNXInputSizeCell);
                                onnxInfoInDLTOrdering(len+1:2) = repmat({1},1,2-len);
                            end
                            UString = string(repmat('U', 1, numel(onnxInfoInDLTOrdering)));
                            textlines = [inputInfoOrderL1, ...
                                getString(message('nnet_cnn_onnx:onnx:UnlabeledInputDescriptionL2')),...
                                "    (" + join(string(onnxInfoInDLTOrdering), ', ') + ")",...
                                getString(message('nnet_cnn_onnx:onnx:UnlabeledInputDescriptionL3', "dlarray(data, '" + UString + "')"))];
                            inputInformation  = join(textlines, newline);
                            createdInputLayer = nnet.onnx.layer.CustomInputLayerMultiOutput(DLTLayerName, DLTTensorName, numel(ONNXInputSizeCell), inputInformation, UString, this.IsInputForwardONNX);
                        end
                    end
                end
            end
        end

        function Layer = createDAGOutputLayer(this, outputTensorName, userClasses, userOutputLayerType)
            if ~isempty(userClasses)
                ClassesNVP = {'Classes', userClasses};
            else
                ClassesNVP = {};
            end
            if ~isempty(userOutputLayerType)
                % Create the kind of layer the user specified. Pass
                % 'Classes' to a classification layer.
                switch userOutputLayerType
                    case {'classification'}
                        Layer = classificationLayer('Name', "ClassificationLayer_" + outputTensorName, ClassesNVP{:});
                    case {'regression'}
                        Layer = regressionLayer('Name', "RegressionLayer_" + outputTensorName);
                    case {'pixelclassification'}
                        Layer = pixelClassificationLayer('Name', "PixelClassificationLayer_" + outputTensorName, ClassesNVP{:});
                    otherwise
                        error(message('nnet_cnn_onnx:onnx:UnsupportedOutputLayerType'));
                end
            else
                % The user did not specify OutputLayerType.
                % Find out if the output tensor was created by a Softmax node
                isFromSoftmax = false;
                nodeName = tensorCreator(this.GraphProtoManager, outputTensorName);
                if ~isempty(nodeName)
                    node = nodesFromNames(this.GraphProtoManager, nodeName);
                    isFromSoftmax = isequal(node.op_type, 'Softmax');
                end
                % Create the right kind of output layer.  Pass 'Classes' to
                % a classification layer.
                fmt = tensorLabel(this.OnnxTensorLabeler, outputTensorName);
                if fmt==""
                    Layer = nnet.onnx.layer.PlaceholderOutputLayer("OutputLayer_" + outputTensorName);
                elseif ismember(fmt, ["BC" "TBC"]) && isFromSoftmax
                    Layer = classificationLayer('Name', "ClassificationLayer_" + outputTensorName, ClassesNVP{:});
                elseif ismember(fmt, ["BCSS" "BSSC" "CSS" "BCSSS" "BSSSC" "CSSS"]) && isFromSoftmax && nnet.internal.cnn.onnx.isInstalledCVST
                    Layer = pixelClassificationLayer('Name', "PixelClassificationLayer_" + outputTensorName, ClassesNVP{:});
                else
                    Layer = regressionLayer('Name', "RegressionLayer_" + outputTensorName);
                end
            end
        end

        function Layer = createOutputPermuteLayer(this, outputTensorName, isDuplicate, duplicateCount)
            Layer = [];
            if isDuplicate
                % Create unique layer name with suffix
                DLTLayerName = nnet.internal.cnn.onnx.makeNNTName(sprintf('%s_%d', outputTensorName, duplicateCount) + "Output");
            else
                DLTLayerName = nnet.internal.cnn.onnx.makeNNTName(outputTensorName + "Output");
            end
            DLTLayerName       = nnet.internal.cnn.onnx.makeNNTName(outputTensorName + "Output");
            outputTensorLabel  = tensorLabel(this.OnnxTensorLabeler, outputTensorName);
            ONNXOutputSizeCell = onnxOutputSizeCell(this.GraphProtoManager, outputTensorName);
            if outputTensorLabel=="" && this.IsInputForwardONNX
                % add a permutation layer at the end of the model if the
                % outputTensorLabel is empty. This condition happens when
                % the last layer is a custom layer. 
                % Add dummy 1's at the end if it's less than 2D
                if numel(ONNXOutputSizeCell)<2
                    Layer = permuteLayer(2:-1:1, 'Name', DLTLayerName);
                else
                    Layer = permuteLayer(numel(ONNXOutputSizeCell):-1:1, 'Name', DLTLayerName);
                end
            elseif isDuplicate
                % For duplicates without format, create identity layer
                Layer = identityLayer('Name', DLTLayerName);
            end
        end

        function minLengthRequired = checkMinLengthRequired(this)
            % checkMinLengthRequired  Check if MinLength has to be set
            lGraph = this.LayerGraph;
            minLengthRequired = false;
            hasPlaceholderLayers = ~isempty(findPlaceholderLayers(lGraph));
            if length(lGraph.InputNames) == 1
                inputLayer = lGraph.Layers(1);
                minLengthRequired   = icheckMinLengthHelper(lGraph,inputLayer, hasPlaceholderLayers);
            end
        end

        function lGraph =  autoSetMinLength(this)
            % autoSetMinLength  Determine MinLength for Sequence Input Layer
            lGraph              = this.LayerGraph;
            inputLayer          = lGraph.Layers(1);
            minLength           = iGetMinLengthForSequenceInput(lGraph);
            inputLayerReplace   = sequenceInputLayer(inputLayer.InputSize,"Name",inputLayer.Name,"MinLength",minLength);
            this.LayerGraph     = replaceLayer(lGraph,inputLayer.Name,inputLayerReplace);
        end

    end
end



function DLTInputSize = iGetDLTImageInputSize(ONNXInputDimCell, inputTensorLabel, userImageInputSize)
% Returns DLTInputSize without the batch dimension, or empty if there are
% any missing entries. inputTensorLabel is any of the formats containing
% images. ONNXInputDimCell is a corresponding cell array of numbers or
% symbols. userImageInputSize is a vector of either [H W C], [H W D C], or
% empty if not specified for this input.
DLTInputSize = [];
% Remove leading t dimension if present
if startsWith(inputTensorLabel, 'T')
    ONNXInputDimCell(1) = [];
    inputTensorLabel = extractAfter(inputTensorLabel, 1);
end
% Remove leading b dimension if present
if startsWith(inputTensorLabel, 'B')
    ONNXInputDimCell(1) = [];
    inputTensorLabel = extractAfter(inputTensorLabel, 1);
end
% Permute to DLT ordering
switch inputTensorLabel
    case {"SSC", "SSSC"}
        DLTInputShapeCell = ONNXInputDimCell;
    case "CSS"
        DLTInputShapeCell = ONNXInputDimCell([2 3 1]);      % css --> ssc
    case "CSSS"
        DLTInputShapeCell = ONNXInputDimCell([2 3 4 1]);    % csss --> sssc
end
% Fill from userImageInputSize if present
if ~isempty(userImageInputSize)
    checkImageInputSizeMatch(DLTInputShapeCell, userImageInputSize)
    DLTInputSize = userImageInputSize;
else
    isMissing = cellfun(@(x) ~isnumeric(x) || isempty(x), DLTInputShapeCell);
    if ~any(isMissing)
        DLTInputSize = cell2mat(DLTInputShapeCell);
    end
end
end

function checkImageInputSizeMatch(ONNXCell, userSize)
for i=1:numel(userSize)
    if ~isempty(ONNXCell{i}) && isnumeric(ONNXCell{i}) && ONNXCell{i}~=userSize(i)
        % A number is present in both and they don't match
        error(message('nnet_cnn_onnx:onnx:ImageInputSizeMismatch', i, userSize(i), ONNXCell{i}));
    end
end
end

function iValidateOpsetVersion(OpsetVersion, maxSupportedOpset, minSupportedOpset)
if OpsetVersion > maxSupportedOpset || OpsetVersion < minSupportedOpset
    nnet.internal.cnn.onnx.util.iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:OpsetVersionUnsupportedForImport', num2str(OpsetVersion), num2str(maxSupportedOpset), num2str(minSupportedOpset));
end
end

function strings = iGetInputConnectionStrings(layer)
strings = layer.Name + "/" + string(layer.InputNames);
end

function strings = iGetOutputConnectionStrings(nonEmptyInput, layer)
if layer.NumOutputs==1
    strings = string(layer.Name);
else
    layerOutputNames = layer.OutputNames(nonEmptyInput);
    for i=1:numel(layerOutputNames)
        strings(i) = string(layer.Name) + "/" + layerOutputNames{i};
    end
end
end

function strings = iMakeInputConnectionStrings(layerName, numInputs)
strings = string.empty;
if numInputs==1
    strings = layerName + "/in";
else
    for i=1:numInputs
        strings(i) = layerName + "/in" + i;
    end
end
end

function strings = iMakeOutputConnectionStrings(node, LayerName)
if numel(node.output)==1
    strings = string(LayerName);
else
    for i=1:numel(node.output)
        strings(i) = string(LayerName) + "/out" + string(i);
    end
end
end

function strings = iMakeCustomOutputConnectionStrings(allOutputNames, LayerName)
% Name all the outputs with their ONNX tensor name
for i=1:numel(allOutputNames)
    strings(i) = string(LayerName) + "/" + allOutputNames(i);
end
end

function sourceString = iWorkaroundSingleOutputSourceIssue(LayerArray, sourceString)
% When a layer has a single output whose name is not "out", then the
% 'Source' table in lg.Connections only shows the layer name, and
% disconnectLayers throws an error because the output port name is not
% "out". The workaround is to append the output name to the source string.
% Needed in addConnectComponentsLayer.
[tf,layerIdx] = ismember(sourceString, {LayerArray.Name});
if tf
    layer = LayerArray(layerIdx);
    if layer.NumOutputs==1 && ~isequal(layer.OutputNames{1}, "out")
        sourceString = sourceString + "/" + layer.OutputNames{1};
    end
end
end

function C = iReplaceEmptyCells(C, newItem)
C(cellfun(@isempty, C)) = {newItem};
end

function nonEmptyNames = iGetNonEmptyNames(tensorCell)
% Remove cell entries containing empty names ('')
emptyStr = cellfun('isempty', tensorCell);
nonEmptyNames = tensorCell(~emptyStr);
end

function minLengthRequired = icheckMinLengthHelper(lGraph,inputLayer, hasPlaceholderLayers)
minLengthRequired = false;
if (any(arrayfun(@(ls)ismember(class(ls),{'nnet.cnn.layer.Convolution1DLayer',...
        'nnet.cnn.layer.MaxPooling1DLayer','nnet.cnn.layer.AveragePooling1DLayer'}), lGraph.Layers)) && ...
        isequal(class(inputLayer), 'nnet.cnn.layer.SequenceInputLayer'))
    if(inputLayer.MinLength == 1)
        if ~hasPlaceholderLayers
            minLengthRequired = true;
        else
            %If LayerGraph has placeholder layers, warn the user about
            %min length being not automatically set
            nnet.internal.cnn.onnx.util.iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:MinLengthNotSet',...
                inputLayer.Name);
        end
    end
end
end

function minLength =  iGetMinLengthForSequenceInput(lGraph)
inputLayer = lGraph.Layers(1);
minLength = 1;
while true
    inputLayerReplace = sequenceInputLayer(inputLayer.InputSize,"Name",inputLayer.Name,"MinLength",minLength);
    lGraph = replaceLayer(lGraph,inputLayer.Name,inputLayerReplace);
    try
        [sizes,formats] = deep.internal.sdk.forwardDataAttributes(lGraph);
        minLength = iBinarySearchMinLength(lGraph,ceil(median(minLength/2:minLength)),floor(minLength/2+1),minLength);
        break;
    catch e
        if iCheckSizeError(e.cause)
            minLength = minLength*2;
            continue;
        else
            minLength = iBinarySearchMinLength(lGraph,ceil(median(minLength/2:minLength)),floor(minLength/2+1),minLength);
            break;
        end
    end
end
end

function minLength =  iBinarySearchMinLength(lGraph,minLength,startValue,endValue)
if startValue == endValue
    return;
else
    inputLayer = lGraph.Layers(1);
    inputLayerReplace = sequenceInputLayer(inputLayer.InputSize,"Name",inputLayer.Name,"MinLength",minLength);
    lGraph = replaceLayer(lGraph,inputLayer.Name,inputLayerReplace);
    try
        [sizes,formats] = deep.internal.sdk.forwardDataAttributes(lGraph);
        endValue = minLength;
        minLength = floor(median(startValue:endValue));
        minLength = iBinarySearchMinLength(lGraph,minLength,startValue,endValue);
    catch e
        if iCheckSizeError(e.cause)
            startValue = minLength+1;
            minLength = floor(median(startValue:endValue));
            minLength = iBinarySearchMinLength(lGraph,minLength,startValue,endValue);

        else
            endValue = minLength;
            minLength = floor(median(startValue:endValue));
            minLength = iBinarySearchMinLength(lGraph,minLength,startValue,endValue);
        end
    end
end
end

function tf = iCheckSizeError(errorCauseList)
tf = any(cellfun(@(cause)ismember(cause.identifier,{'nnet_cnn:internal:cnn:layer:Convolution1D:FilterSizeLargerThanInput',...
    'nnet_cnn:internal:cnn:layer:MaxPooling1D:PoolSizeLargerThanInput',...
    'nnet_cnn:internal:cnn:layer:AveragePooling1D:PoolSizeLargerThanInput'}),errorCauseList));
end

function tf = iIsInvalidSize(x)
tf = isempty(x) || ~isnumeric(x) || (isnumeric(x) && x <= 0);
end

function dim = iDLTVariableBatchSizeDim(inputLayer)
% If inputLayer takes a variable batch size, return the dimension number in
% its output tensor that represents batchsize, otherwise return NaN. This
% function must return non-Nan for all input layers that the importer may
% put into the returned network, and which have a variable batch size in a
% known location.
if isa(inputLayer, 'nnet.cnn.layer.FeatureInputLayer')
    dim = 2;        % CB
elseif isa(inputLayer, 'nnet.cnn.layer.SequenceInputLayer')
    dim = numel(inputLayer.InputSize) + 1;   % CBT, SSCBT, SSSCBT
elseif isa(inputLayer, 'nnet.cnn.layer.ImageInputLayer')
    dim = 4;        % SSCB
elseif isa(inputLayer, 'nnet.cnn.layer.Image3DInputLayer')
    dim = 5;        % SSSCB
elseif isa(inputLayer, 'nnet.onnx.layer.CustomInputLayer')
    dim = strfind(L.DLTTensorFormat,'B');
    if isempty(dim)
        dim = NaN;
    end
else
    dim = NaN;
end
end
