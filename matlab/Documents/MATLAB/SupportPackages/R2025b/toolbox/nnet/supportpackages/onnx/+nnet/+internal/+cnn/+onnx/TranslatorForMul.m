classdef TranslatorForMul < nnet.internal.cnn.onnx.OperatorTranslator

    % Copyright 2021-2024 The MathWorks, Inc.

    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
        end

        % Given the node and the current input and output formats, update
        % the formats in the specified direction to the extent possible.
        % Direction is either "forward" or "reverse". Each format variable
        % is a string array, where each string is one of the members of the
        % constant 'SupportedONNXLabels' property, or empty. Set a format
        % string to empty if its format cannot be determined.
        function [inputFormats, outputFormats] = propagateTensorFormats(this, direction, inputFormats, outputFormats)
            [inputFormats, outputFormats] = propagateMISOBroadcastOp(this, direction, inputFormats, outputFormats);
        end

        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            LayerName   = this.Node.name;
            if this.GenerateCustomLayers==false && ~any(isNodeInputInitializer(this.GraphProtoManager, this.Node, 1:2))
                % Neither input is an initializer and we're not generating
                % custom layers. So it's ok if we don't know the tensor
                % formats.
                [Layer, issues] = constructLayer(this, 'multiplicationLayer', LayerName, this.Node, 2, 'Name', LayerName);
                return;
            end
            if outputTensorFormats(1)~=""
                % Output must be labeled to be supported
                if all(inputTensorFormats~="") && ~any(isNodeInputInitializer(this.GraphProtoManager, this.Node, 1:2))
                    % Both inputs are labeled and neither is an
                    % initializer.
                    [Layer, issues] = constructLayer(this, 'multiplicationLayer', LayerName, this.Node, 2, 'Name', LayerName);
                elseif (inputTensorFormats(1)~="" && isNodeInputInitializer(this.GraphProtoManager, this.Node, 2) || ...  	% A*const
                        inputTensorFormats(2)~="" && isNodeInputInitializer(this.GraphProtoManager, this.Node, 1))          % const*B
                    % Exactly one input is a constant. Make an ScalingLayer.
                    isParam     = isTensorInitializer(this.GraphProtoManager, this.Node.input);
                    constIdx    = find(isParam,1);
                    constName   = this.Node.input{constIdx};
                    constDim    = initializerSize(this.GraphProtoManager, constName);
                    rawConstData = single(initializerRawData(this.GraphProtoManager, constName));
                    knownFormat = inputTensorFormats(3-constIdx);
                    constData   = permuteConstInput(this, rawConstData, constDim, knownFormat);
                    [Layer, issues] = constructLayer(this, 'scalingLayer', LayerName, ...
                        this.Node, 'Name', LayerName, 'Scale', constData, 'Offset', 0);
                end
            end
        end
    end
end