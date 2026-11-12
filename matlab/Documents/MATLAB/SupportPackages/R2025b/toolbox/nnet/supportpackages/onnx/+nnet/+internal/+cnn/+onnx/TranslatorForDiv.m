classdef TranslatorForDiv < nnet.internal.cnn.onnx.OperatorTranslator
    % A class to translate ONNX Div operators into MATLAB layers
    
    % Copyright 2021-2024 The MathWorks, Inc.
    properties(SetAccess = protected)
        LayerName
    end
    
    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            % Set the layer name
            this.LayerName = this.Node.name;
        end
        
        % Given the node and the current input and output formats, update
        % the formats in the specified direction to the extent possible.
        % Direction is either "forward" or "reverse". Each format variable
        % is a string array, where each string is one of the members of the
        % constant 'SupportedONNXLabels' property, or empty. Set a format
        % string to empty if its format cannot be determined.
        function [inputTensorFormats, outputTensorFormats] = propagateTensorFormats(this, direction, inputTensorFormats, outputTensorFormats)
            [inputTensorFormats, outputTensorFormats] = propagateMISOBroadcastOp(this, direction, inputTensorFormats, outputTensorFormats);
        end
        
        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            % Div is fully supported if the first input and output are
            % labeled, and the second input is an initializer.
            if inputTensorFormats(1)~="" && outputTensorFormats(1)~="" ...
                    && isNodeInputInitializer(this.GraphProtoManager, this.Node, 2)
                % Make an ElementwiseAffineLayer.
                constName   = this.Node.input{2};
                const       = single(initializerRawData(this.GraphProtoManager, constName));
                constDim    = initializerSize(this.GraphProtoManager, constName);
                knownFormat = inputTensorFormats(1);
                const       = permuteConstInput(this, const, constDim, knownFormat);
                % Create layer
                [Layer, issues] = constructLayer(this, 'scalingLayer', this.LayerName, ...
                    this.Node, 'Name', this.LayerName, 'Scale', 1./const, 'Offset', 0);
            end
        end
    end
end