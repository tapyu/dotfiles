classdef TranslatorForReduceOperators < nnet.internal.cnn.onnx.OperatorTranslator
    %   Copyright 2021-2024 The MathWorks, Inc.    

    properties(SetAccess = protected)
        % An attribute for ReduceSum operator with Opset<13
        %              for all other Reduce{ops} operators with Opset<18
        Axes        
        keepdims
        noop_with_empty_axes
    end

    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            if isAxesAttr(this)
                AttributeTable = cell2table({
                    "axes"     	"INTS"	true    0
                    "keepdims"	"INT"  	true    1
                    });
                [this.Axes, this.keepdims] = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
            else
                % Opset >= 18
                AttributeTable = cell2table({
                    "keepdims"	            "INT"  	true    1
                    "noop_with_empty_axes"	"INT"  	true    0
                    });
                [this.keepdims, this.noop_with_empty_axes] = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
            end
        end

        % Given the node and the current input and output formats, update
        % the formats in the specified direction to the extent possible.
        % Direction is either "forward" or "reverse". Each format variable
        % is a string array, where each string is one of the members of the
        % constant 'SupportedONNXLabels' property, or empty. Set a format
        % string to "" if its format cannot be determined.
        function [inputFormats, outputFormats] = propagateTensorFormats(this, direction, inputFormats, outputFormats)
            if isAxesAttr(this)
                % Opset<13: ReduceSum acts just like all ther other Reduce ops
                if this.keepdims==1
                    [inputFormats, outputFormats] = propagateSISOPassthroughOp(this, direction, inputFormats, outputFormats);
                else
                    if direction=="forward" && inputFormats~="" && outputFormats==""
                        % Do the SISO propagation and then do a squeeze18
                        % propagation. Perform the squeeze on the format string
                        % and see if the resulting format is supported. We do
                        % not check that the squeezed dimensions are
                        % singletons. If they were not, it would not have been
                        % a valid ONNX file.
                        [~, tempOutputFormats] = propagateSISOPassthroughOp(this, direction, inputFormats, outputFormats);
                        in = inputFormats(1);
                        if ~isempty(this.Axes) % Empty axes means squeeze all singleton dimensions
                            numDimsX = strlength(in);
                            axes = this.Axes;
                            axes(axes<0) = axes(axes<0) + numDimsX;
                            inchar = char(in);
                            inchar(axes+1) = [];
                            newout = string(inchar);
                            if ismember(newout, this.SupportedONNXLabels)
                                tempOutputFormats(1) = newout;
                                outputFormats = tempOutputFormats;
                            end
                        end
                    end
                end
            else
                % Opset>=13: ReduceSum takes Axes as an input. Since we
                % don't know the value of Axes yet, we cannot remove the
                % axes and propagate through the squeeze here as we did for
                % Opset<13, so we only propagate formats when keepdims==1.
                % When keepdims==1, the value of noop_with_empty_axes is
                % irrelevant so we can ignore it.
                if this.keepdims==1
                    [inputFormats, outputFormats] = propagateSISOPassthroughOp(this, direction, inputFormats, outputFormats);
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

    methods(Access=protected)
        function tf = isAxesAttr(this)
            tf = false;
            optVersion = this.GraphProtoManager.OpsetVersion;
            if isequal(this.Node.op_type, 'ReduceSum') && optVersion < 13 ||...
                    ~isequal(this.Node.op_type, 'ReduceSum') && optVersion < 18
                tf = true;
            end
        end
    end
end

