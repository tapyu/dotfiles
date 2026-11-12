classdef Opset9TranslationStrategy < nnet.internal.cnn.onnx.fcn.matlab.Opset8TranslationStrategy
 % A strategy class implementing operator translators for opset version
    % 9.
 
%   Copyright 2020-2023 The MathWorks, Inc.     
    methods
        nodeTranslation = translateBatchNormalization(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateCompress(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateConstantOfShape(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateErf(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateNonZero(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateOneHot(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateScan(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateScatter(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateSign(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateUpsample(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateWhere(~, nodeTranslation, node, IntegerTensorNames);
    end
end
