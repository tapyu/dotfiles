classdef TranslatorForSub < nnet.internal.cnn.onnx.OperatorTranslator

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
            % Sub is fully supported if exactly one input is an
            % initializer, the other input is labeled, and the output is
            % labeled. That is, we have C=A-const or C=const-B.
            if outputTensorFormats(1)~="" && ...
                    (inputTensorFormats(1)~="" && isNodeInputInitializer(this.GraphProtoManager, this.Node, 2) || ...   % A-const
                    inputTensorFormats(2)~="" && isNodeInputInitializer(this.GraphProtoManager, this.Node, 1))         % const-B
                % One input is a constant. Make an ElementwiseAffineLayer.
                LayerName   = this.Node.name;
                isParam     = isTensorInitializer(this.GraphProtoManager, this.Node.input);
                constIdx    = find(isParam,1);
                constName   = this.Node.input{constIdx};
                constDim    = initializerSize(this.GraphProtoManager, constName);
                rawConstData = single(initializerRawData(this.GraphProtoManager, constName));
                knownFormat = inputTensorFormats(3-constIdx);
                constData   = permuteConstInput(this, rawConstData, constDim, knownFormat);
                % Create layer.
                if constIdx==2
                    % A-const case. Create affine(A,1,-const) = 1*A-const
                    [Layer, issues] = constructLayer(this, 'scalingLayer', ...
                    LayerName, this.Node, 'Name', LayerName,...
                    'Scale', 1, 'Offset',-constData);
                else
                    % const-B case. Create affine(B,-1,const) = -1*B+const
                    [Layer, issues] = constructLayer(this, 'scalingLayer', ...
                    LayerName, this.Node, 'Name', LayerName,...
                    'Scale', -1, 'Offset',constData);
                end
            end
        end
    end
end