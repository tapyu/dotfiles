classdef TranslatorForConcat < nnet.internal.cnn.onnx.OperatorTranslator

    %   Copyright 2021-2024 The MathWorks, Inc.    

    properties
        axis
    end
    
    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
            % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
            AttributeTable = cell2table({
                "axis"        "INT"       true    1
                });
            this.axis = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
        end
        
        % Given the node and the current input and output formats, update
        % the formats in the specified direction to the extent possible.
        % Direction is either "forward" or "reverse". Each format variable
        % is a string array, where each string is one of the members of the
        % constant 'SupportedONNXLabels' property, or empty. Set a format
        % string to "" if its format cannot be determined.
        function [inputFormats, outputFormats] = propagateTensorFormats(this, direction, inputFormats, outputFormats)
            [inputFormats, outputFormats] = propagateMISOPassthroughOp(this, direction, inputFormats, outputFormats);
        end
        
        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            % Supported if (1) all formats are known, (2) we're not
            % concatenating over the batch dimension, (3) no inputs are
            % initializers
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            if any([inputTensorFormats outputTensorFormats] == "")
                supported = false;
            elseif isConcatOverBatchDim(this, outputTensorFormats) ...
                    && hasMultipleInputs(this) % If there's only one input, we create an Identity layer, so the axis does not matter. 
                supported = false;
                issues = nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                    message('nnet_cnn_onnx:onnx:AxisIsBatchDim', 'Concat'));
            else
                supported = ~any(isNodeInputInitializer(this.GraphProtoManager, this.Node, 1:numel(this.Node.input)));
            end
            if supported
                LayerName = this.Node.name;
                % If there is only one input, return an identity layer.
                if numel(this.Node.input) == 1
                    % If the user is importing into a dlnetwork, create
                    % identityLayer (which is only supported in dlnetwork).
                    % If the user has requested a DAGNetwork, then create
                    % the legacy nnet.onnx.layer.IdentityLayer.
                    if this.TargetNetwork=="dlnetwork"
                        Layer = identityLayer('Name',LayerName);
                    else % DAGNetwork
                        Layer = nnet.onnx.layer.IdentityLayer(LayerName);
                    end
                else
                    % Create concatenationLayer using the converted axis
                    DLTAxis = getDLTAxis(this, outputTensorFormats);
                    [Layer, constructionIssue] = constructLayer(this, 'concatenationLayer', LayerName, this.Node, DLTAxis, numel(this.Node.input), 'Name', LayerName);
                    issues = [issues constructionIssue];
                end
            end
        end
    end
    
    methods(Access=private)
        function DLTAxis = getDLTAxis(this, onnxFormat)
            % Get the DLT axis corresponding to the onnxAxis for the given onnxFormat.
            onnxAxis = this.axis;
            if onnxAxis < 0
                onnxAxis = onnxAxis + strlength(onnxFormat);
            end
            permToONNX = nnet.internal.cnn.onnx.ModelTranslationIntoLayers.permDLTToONNX(onnxFormat);
            DLTAxis = permToONNX(onnxAxis+1);
        end
        
        function tf = isConcatOverBatchDim(this, onnxFormat)
            % Return true if the ONNX op is concatenting over the batch dim
            onnxAxis = this.axis;
            if onnxAxis < 0
                onnxAxis = onnxAxis + strlength(onnxFormat);
            end
            batchPos = strfind(onnxFormat, "B");
            tf = isequal(batchPos, onnxAxis+1);
        end

        function tf = hasMultipleInputs(this)
            tf = numel(this.Node.input) > 1;
            if tf
                tf = any(cellfun(@(x) ~isempty(x), this.Node.input(2:end)));
            end
        end
    end
end

