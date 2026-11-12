classdef TranslatorForElu < nnet.internal.cnn.onnx.OperatorTranslator
    % A class to translate ONNX Elu operators into MATLAB layers
    
    % Copyright 2021 The MathWorks, Inc.
    
    properties(SetAccess = protected)
        % Operator attributes
        alpha
        
        % Other properties
        LayerName
    end
    
    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            this.CanPropagateSingletonFormats = true;

            % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
            % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
            AttributeTable = cell2table({
                "alpha"      "FLOAT"    true    1
                });
            % Parse the attributes
            this.alpha = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
            
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
            if isLabeled(this, inputTensorFormats, outputTensorFormats) || this.GenerateCustomLayers==false
                [Layer, issues] = constructLayer(this, 'nnet.cnn.layer.ELULayer', this.LayerName, this.Node, this.alpha, this.LayerName);
            end
        end
    end
    
    methods(Access=protected)
        function tf = isLabeled(~, inputTensorFormats, outputTensorFormats)
            tf = ~any(strcmp(inputTensorFormats, "")) && ~any(strcmp(outputTensorFormats, ""));
        end
        
    end
end
