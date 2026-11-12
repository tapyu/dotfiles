classdef TranslatorForSoftmax < nnet.internal.cnn.onnx.OperatorTranslator

    %   Copyright 2021 The MathWorks, Inc.    

    properties
        axis
    end
    
    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            this.CanPropagateSingletonFormats = true;

            if this.GraphProtoManager.OpsetVersion <= 12
                defaultAxis = 1;
            else
                defaultAxis = -1;
            end
            AttributeTable = cell2table({
                "axis"        "INT"       true    defaultAxis
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
            [inputFormats, outputFormats] = propagateSISOPassthroughOp(this, direction, inputFormats, outputFormats);
        end
        
        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            if all([inputTensorFormats outputTensorFormats] ~= "") && ...
                    isAxisChannelDimension(this, outputTensorFormats)
                [Layer, issues] = constructLayer(this, 'softmaxLayer', this.Node.name, this.Node, 'Name', this.Node.name);
            end
        end
    end
    
    methods(Access=private)
        function tf = isAxisChannelDimension(this, onnxFormat)
            % Return true if the ONNX axis indicates the channel dimension
            onnxAxis = this.axis;
            if onnxAxis < 0
                onnxAxis = onnxAxis + strlength(onnxFormat);
            end
            channelPos = strfind(onnxFormat, "C");
            tf = isequal(channelPos, onnxAxis+1);
        end
    end
end
