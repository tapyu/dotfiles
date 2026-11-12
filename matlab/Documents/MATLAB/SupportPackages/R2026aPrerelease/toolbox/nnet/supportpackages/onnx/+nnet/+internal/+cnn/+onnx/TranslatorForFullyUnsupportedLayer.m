classdef TranslatorForFullyUnsupportedLayer < nnet.internal.cnn.onnx.OperatorTranslator
    %   Copyright 2021 The MathWorks, Inc.    

    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
        end
        
        function [inputFormats, outputFormats] = propagateTensorFormats(this, direction, inputFormats, outputFormats)
            % No propagation is possible.
        end
        
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            % No layer.
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
        end
    end
end
