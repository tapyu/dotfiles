classdef Opset7TranslationStrategy < nnet.internal.cnn.onnx.fcn.coder.Opset6TranslationStrategy
    % A strategy class implementing operator translators for opset version
    % 7.

    %   Copyright 2024-2025 The MathWorks, Inc.
    methods
        nodeTranslation = translateAcos(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateAdd(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateAnd(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateCos(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateDiv(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateEqual(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateGreater(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateLess(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateMul(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateOr(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translatePow(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateSin(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateSub(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateTan(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateUpsample(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateXor(this, nodeTranslation, node, ~);
        nodeTranslation = translateInfixBinaryOperator(~, nodeTranslation, node, MATLABFcn, IntegerTensorNames);
        nodeTranslation = translatePrefixBinaryOperator(~, nodeTranslation, node, MATLABFcn, ~);     
    end
end
