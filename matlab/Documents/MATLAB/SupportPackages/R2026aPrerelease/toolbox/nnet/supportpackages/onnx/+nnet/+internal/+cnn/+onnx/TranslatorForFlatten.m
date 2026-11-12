classdef TranslatorForFlatten < nnet.internal.cnn.onnx.OperatorTranslator
    % A class to translate ONNX Flatten operators into MATLAB layers

    % Copyright 2021 The MathWorks, Inc.

    properties(SetAccess=protected)
        Axis
    end

    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            AttributeTable = cell2table({
                "axis"     "INT"      true    1
                });
            % Parse the attributes
            this.Axis = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
        end

        % Given the node and the current input and output formats, update
        % the formats in the specified direction to the extent possible.
        % Direction is either "forward" or "reverse". Each format variable
        % is a string array, where each string is one of the members of the
        % constant 'SupportedONNXLabels' property, or empty. Set a format
        % string to "" if its format cannot be determined.
        function [inputFormats, outputFormats] = propagateTensorFormats(this, direction, inputFormats, outputFormats)
            % We can do forward propagation in the following case: The
            % input has a known label that starts with "B", and axis=1. In
            % this case, we will propagate forward to "BC".
            if direction=="forward" && outputFormats(1)=="" && startsWith(inputFormats(1),"B")
                numDimsX = strlength(inputFormats(1));
                axis = this.Axis;
                axis(axis<0) = axis(axis<0) + numDimsX;
                if axis==1
                    % Propagate "Bxxx..." to BC
                    outputFormats(1) = "BC";
                end
            end
        end

        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            % BCSS --> BC
            if inputTensorFormats=="BCSS" && outputTensorFormats=="BC"
                [Layer, constructionIssue] = constructLayer(this, 'nnet.onnx.layer.FlattenInto2dLayer', this.Node.name, this.Node, this.Node.name);
                issues = [issues constructionIssue];
            end
            % BCSSS --> BC
            if inputTensorFormats=="BCSSS" && outputTensorFormats=="BC"
                [Layer, constructionIssue] = constructLayer(this, 'nnet.onnx.layer.Flatten3dInto2dLayer', this.Node.name, this.Node, this.Node.name);
                issues = [issues constructionIssue];
            end
        end
    end
end
