classdef Opset10TranslationStrategy < nnet.internal.cnn.onnx.fcn.coder.Opset9TranslationStrategy
    % A strategy class implementing operator translators for opset version
    % 10.
    
    %   Copyright 2024 The MathWorks, Inc.
    methods
        nodeTranslation = translateAveragePool(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateMaxPool(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateMod(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateNonMaxSuppression(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateResize(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateRoiAlign(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateSlice(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateTopK(~, nodeTranslation, node, IntegerTensorNames);
    end
end
