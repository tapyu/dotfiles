classdef TranslatorForUnsqueeze < nnet.internal.cnn.onnx.TranslatorForFullyUnsupportedLayer
    
    % Copyright 2021-2024 The MathWorks, Inc.
    
    properties(SetAccess=protected)
        Axes
        Opset
    end
    
    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            this.CanPropagateSingletonFormats = true;
            this.Opset = this.GraphProtoManager.OpsetVersion;
            this.Axes = [];
            if this.Opset < 13
                AttributeTable = cell2table({
                    "axes"     "INTS"      true    []
                });
                % Parse the attributes
                this.Axes = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
            end
            
        end
        
        % Given the node and the current input and output formats, update
        % the formats in the specified direction to the extent possible.
        % Direction is either "forward" or "reverse". Each format variable
        % is a string array, where each string is one of the members of the
        % constant 'SupportedONNXLabels' property, or empty. Set a format
        % string to "" if its format cannot be determined.
        function [inputFormats, outputFormats] = propagateTensorFormats(this, direction, inputFormats, outputFormats)
            in = inputFormats(1);
            out = outputFormats(1);
            if direction=="forward"
                % Can't propagate forward through an unsqueeze.
            elseif in=="" && out~="" && this.Opset < 13
                % backward.
                % Perform a squeeze on the output format string and see
                % if the result is supported
                numDimsY = strlength(out);
                axes = this.Axes;
                axes(axes<0) = axes(axes<0) + numDimsY;
                outchar = char(out);
                outchar(axes+1) = [];
                newin = string(outchar);
                if ismember(newin, this.SupportedONNXLabels)
                    inputFormats(1) = newin;
                end
            end
        end
        
        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
        end
    end
end
