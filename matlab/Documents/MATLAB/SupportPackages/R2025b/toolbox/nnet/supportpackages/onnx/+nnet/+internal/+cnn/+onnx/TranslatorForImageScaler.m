classdef TranslatorForImageScaler < nnet.internal.cnn.onnx.OperatorTranslator
    % A class to translate ONNX ImageScaler operators into MATLAB layers
    
    % Copyright 2021-2024 The MathWorks, Inc.

    properties(SetAccess = protected)
        % Operator attributes
        bias
        scale
        
        % Other properties
        LayerName
    end
    
    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
            % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
            AttributeTable = cell2table({
                "bias"          "FLOATS"   true    1
                "scale"         "FLOAT"    true    1
                });
            [this.bias, this.scale] = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
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
            if ismember(inputTensorFormats(1), ["BCSS", "BCSSS"]) && ismember(outputTensorFormats(1), ["BCSS", "BCSSS"])
                if isequal(inputTensorFormats(1), "BCSSS")
                    [Layer, issues] = constructLayer(this, 'scalingLayer', this.LayerName, this.Node, ...
                        'Name', this.LayerName, 'Scale', this.scale, 'Offset', reshape(this.bias, [1 1 1 numel(this.bias)]));
                else % BCSS
                    [Layer, issues] = constructLayer(this, 'scalingLayer', this.LayerName, this.Node, ...
                        'Name', this.LayerName, 'Scale', this.scale, 'Offset', reshape(this.bias, [1 1 numel(this.bias)]));
                end
            end
            
        end
    end
end

