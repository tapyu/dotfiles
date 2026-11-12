classdef TranslatorForSqueeze < nnet.internal.cnn.onnx.OperatorTranslator
    
    %   Copyright 2021-2024 The MathWorks, Inc.

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
            % Perform the squeeze on the format string and see if the
            % resulting format is supported. We do not check that the
            % squeezed dimensions are singletons. If they were not, it
            % would not have been a valid ONNX file.
            in  = inputFormats(1);
            out = outputFormats(1);
            if this.Opset<13 && direction=="forward" && ~isempty(this.Axes) % Empty axes means squeeze all singleton dimensions
                if in~="" && out==""
                    numDimsX = strlength(in);
                    axes = this.Axes;
                    axes(axes<0) = axes(axes<0) + numDimsX;
                    inchar = char(in);
                    inchar(axes+1) = [];
                    newout = string(inchar);
                    if ismember(newout, this.SupportedONNXLabels)
                        outputFormats(1) = newout;
                    end
                end
            else
                % backward. Potentially could go backward thru 1BC-->BC and
                % T1BC-->tbc. Left for future enhancement.
            end
        end
        
        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            % BCSS --> BC
            if inputTensorFormats(1)=="BCSS" && outputTensorFormats=="BC"
                [Layer, constructionIssue] = constructLayer(this, 'nnet.onnx.layer.FlattenInto2dLayer', this.Node.name, this.Node, this.Node.name);
                issues = [issues constructionIssue];
            end
            % BCSSS --> BC
            if inputTensorFormats(1)=="BCSSS" && outputTensorFormats=="BC"
                [Layer, constructionIssue] = constructLayer(this, 'nnet.onnx.layer.Flatten3dInto2dLayer', this.Node.name, this.Node, this.Node.name);
                issues = [issues constructionIssue];
            end
        end
    end
end
