classdef TranslatorForArgMinMax < nnet.internal.cnn.onnx.OperatorTranslator

    %   Copyright 2021 The MathWorks, Inc.    

    properties(SetAccess = protected)
        Axis
        keepdims
    end
    
    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            AttributeTable = cell2table({
                "axis"     	"INT"	true    0
                "keepdims"	"INT"  	true    1
                });
            [this.Axis, this.keepdims] = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
        end
        
        % Given the node and the current input and output formats, update
        % the formats in the specified direction to the extent possible.
        % Direction is either "forward" or "reverse". Each format variable
        % is a string array, where each string is one of the members of the
        % constant 'SupportedONNXLabels' property, or empty. Set a format
        % string to "" if its format cannot be determined.
        function [inputFormats, outputFormats] = propagateTensorFormats(this, direction, inputFormats, outputFormats)
            if this.keepdims==1
                [inputFormats, outputFormats] = propagateSISOPassthroughOp(this, direction, inputFormats, outputFormats);
            else
                if direction=="forward" && inputFormats~="" && outputFormats==""
                    % Do the SISO propagation and then do a squeeze
                    % propagation. Perform the squeeze on the format string
                    % and see if the resulting format is supported. We do
                    % not check that the squeezed dimension is a singleton.
                    % If not, it would not have been a valid ONNX file.
                    [~, tempOutputFormats] = propagateSISOPassthroughOp(this, direction, inputFormats, outputFormats);
                    in = inputFormats(1);
                    numDimsX = strlength(in);
                    axis = this.Axis;
                    axis(axis<0) = axis(axis<0) + numDimsX;
                    inchar = char(in);
                    inchar(axis+1) = [];
                    newout = string(inchar);
                    if ismember(newout, this.SupportedONNXLabels)
                        tempOutputFormats(1) = newout;
                        outputFormats = tempOutputFormats;
                    end
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
