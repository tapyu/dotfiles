classdef TranslatorForAdd < nnet.internal.cnn.onnx.OperatorTranslator

    %   Copyright 2021-2024 The MathWorks, Inc.    

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
            % "Add" is fully supported if all input and output tensors are
            % labeled, or if the output is labeled, only one input is
            % labeled and the other is an initializer.
            if outputTensorFormats==""
                supported = false;
            else
                supported = all(inputTensorFormats~="") || ...
                    ((inputTensorFormats(1)~="" && isNodeInputInitializer(this.GraphProtoManager, this.Node, 2)) || ...
                    (inputTensorFormats(2)~="" && isNodeInputInitializer(this.GraphProtoManager, this.Node, 1)));
            end
            if supported
                LayerName = this.Node.name;
                % Add takes 2 inputs.
                isParam = isTensorInitializer(this.GraphProtoManager, this.Node.input);
                if ~any(isParam)
                    % All inputs are from layers. Make an additionLayer.
                    [Layer, issues] = constructLayer(this, 'additionLayer', LayerName, this.Node, numel(this.Node.input), 'Name', LayerName);
                else
                    % One input is a constant. Make an ScalingLayer.
                    paramIdx    = find(isParam,1);
                    constName   = this.Node.input{paramIdx};
                    constDim    = initializerSize(this.GraphProtoManager, constName);
                    knownFormat = inputTensorFormats(3-paramIdx);
                    rawData  	= single(initializerRawData(this.GraphProtoManager, constName));
                    constData   = permuteConstInput(this, rawData, constDim, knownFormat);
                    
                    % Create layer
                    [Layer, issues] = constructLayer(this, 'scalingLayer',...
                        LayerName, this.Node, 'Name', LayerName, 'Scale', 1, 'Offset', constData);
                end
            end
        end
    end
end

