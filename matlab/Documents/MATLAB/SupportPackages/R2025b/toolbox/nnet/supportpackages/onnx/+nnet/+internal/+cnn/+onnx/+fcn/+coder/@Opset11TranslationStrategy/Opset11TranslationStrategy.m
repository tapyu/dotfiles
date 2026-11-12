classdef Opset11TranslationStrategy < nnet.internal.cnn.onnx.fcn.coder.Opset10TranslationStrategy
    % A strategy class implementing operator translators for opset version
    % 11.

    %   Copyright 2024 The MathWorks, Inc.
    methods
        nodeTranslation = translateClip(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateCompress(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateConstant(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateCumSum(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateDepthToSpace(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateGatherElements(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateGatherND(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateGemm(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translatePad(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateRange(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateResize(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateRound(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateScatterElements(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateScatterND(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateSequenceAt(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateSplitToSequence(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateTopK(~, nodeTranslation, node, IntegerTensorNames);
    end
end
