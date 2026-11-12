classdef TranslatorForNeg < nnet.internal.cnn.onnx.OperatorTranslator

    %   Copyright 2021-2024 The MathWorks, Inc.    

    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            this.CanPropagateSingletonFormats = true;
        end
        
        % Given the node and the current input and output formats, update
        % the formats in the specified direction to the extent possible.
        % Direction is either "forward" or "reverse". Each format variable
        % is a string array, where each string is one of the members of the
        % constant 'SupportedONNXLabels' property, or empty. Set a format
        % string to empty if its format cannot be determined.
        function [inputFormats, outputFormats] = propagateTensorFormats(this, direction, inputFormats, outputFormats)
            [inputFormats, outputFormats] = propagateSISOPassthroughOp(this, direction, inputFormats, outputFormats);
        end
        
        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            if inputTensorFormats(1)~="" || this.GenerateCustomLayers==false
                [Layer, issues] = constructLayer(this, 'scalingLayer', this.Node.name, this.Node, ...
                    'Name', this.Node.name, 'Scale', -1, 'Offset', 0);
            end
        end
    end
end