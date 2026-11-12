classdef TranslatorForDepthToSpace < nnet.internal.cnn.onnx.OperatorTranslator
    % A class to translate ONNX DepthToSpace operators into MATLAB layers
    
    % Copyright 2021 The MathWorks, Inc.
    
    properties(SetAccess = protected)
        % Operator attributes
        blocksize
        mode
        
        % Other properties
        LayerName
    end
    
    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
            % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
            AttributeTable = cell2table({
                "blocksize"          "INT"    false    []
                "mode"	       "STRING"   true    "dcr"
                });
            [this.blocksize, this.mode] = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
            % Set the layer name
            this.LayerName = this.Node.name;
        end
        
        % Given the node and the current input and output formats, update
        % the formats in the specified direction to the extent possible.
        % Direction is either "forward" or "reverse". Each format variable
        % is a string array, where each string is one of the members of the
        % constant 'SupportedONNXLabels' property, or empty. Set a format
        % string to empty if its format cannot be determined.
        function [inputTensorFormats, outputTensorFormats] = propagateTensorFormats(this, direction, inputTensorFormats, outputTensorFormats)
            [inputTensorFormats, outputTensorFormats] = propagateSISOPassthroughOp(this, direction, inputTensorFormats, outputTensorFormats);
        end
        
        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            if ~nnet.internal.cnn.onnx.isInstalledIPT()
                issues = nnet.internal.cnn.onnx.NodeTranslationError(this.Node, message('nnet_cnn_onnx:onnx:noIPTForOperator', 'DepthToSpace'));
            elseif isequal(inputTensorFormats(1), "BCSS") ...
                    && isequal(outputTensorFormats(1), "BCSS")
                [Layer, issues] = constructLayer(this, "depthToSpace2dLayer", this.LayerName, this.Node, this.blocksize, 'Mode', this.mode,'Name', this.LayerName);
            end
        end
    end
end
