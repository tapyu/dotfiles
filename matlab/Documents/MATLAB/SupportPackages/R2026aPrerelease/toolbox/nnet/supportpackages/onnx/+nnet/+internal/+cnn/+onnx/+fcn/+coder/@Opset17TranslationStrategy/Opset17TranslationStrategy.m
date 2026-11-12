classdef Opset17TranslationStrategy < nnet.internal.cnn.onnx.fcn.coder.Opset16TranslationStrategy
%
% A strategy class implementing operator translators for opset version
% 17.

%   Copyright 2024 The MathWorks, Inc. 
    methods
        % Translators
        nodeTranslation = translateLayerNormalization(~, nodeTranslation, node, ~);

    end
end