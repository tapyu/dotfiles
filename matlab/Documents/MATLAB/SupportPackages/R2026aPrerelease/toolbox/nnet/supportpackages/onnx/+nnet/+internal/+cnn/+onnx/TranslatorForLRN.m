classdef TranslatorForLRN < nnet.internal.cnn.onnx.OperatorTranslator
    % A class to translate ONNX LRN operators into MATLAB layers
    
    % Copyright 2021 The MathWorks, Inc.
    
    properties(SetAccess = protected)
        % Operator attributes
        alpha
        beta
        bias
        size
        
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
                "alpha"    "FLOAT"      true    .0001
                "beta"     "FLOAT"  	true    .75
                "bias"     "FLOAT"      true    1
                "size"     "INT"        false   []
                });
            % Parse the attributes
            [this.alpha, this.beta, this.bias, this.size] = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
            
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
            if supportedFormat(this, inputTensorFormats, outputTensorFormats)
                [Layer, issues] = constructLayer(this, 'crossChannelNormalizationLayer', ...
                    this.LayerName, this.Node, this.size, 'Alpha', this.alpha, 'Beta', this.beta, 'K', this.bias, 'Name', this.LayerName);
            end
        end
    end
    
    methods(Access=protected)
        function tf = supportedFormat(~, inputTensorFormat, outputTensorFormat)
            % To be fully supported, the input and output format must both
            % be "BCSS", or "BCSSS".
            supportedFormats = ["BCSS", "BCSSS"];
            tf = ismember(inputTensorFormat(1), supportedFormats) && ...
                isequal(inputTensorFormat(1), outputTensorFormat);
        end
    end
end
