classdef Opset6TranslationStrategy
    % A strategy class implementing operator translators for opset version
    % 6.

    %   Copyright 2024-2025 The MathWorks, Inc. 
    methods
        % Translation utilities 
        C = iStringifyEmpties(this, C);
        s = iTranslateActivationFunction(this, s);

        % Translators
        nodeTranslation = translateAbs(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateAdd(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateAnd(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateArgMax(this, nodeTranslation, node, ~);
        nodeTranslation = translateAveragePool(~, nodeTranslation, node, ~);
        nodeTranslation = translateBatchNormalization(~, nodeTranslation, node, ~);
        nodeTranslation = translateCast(~, nodeTranslation, node, ~);
        nodeTranslation = translateCategoryMapper(~, nodeTranslation, node, ~);
        nodeTranslation = translateCeil(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateClip(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateConcat(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateConstant(~, nodeTranslation, node, ~);
        nodeTranslation = translateConv(~, nodeTranslation, node, ~);
        nodeTranslation = translateConvTranspose(~, nodeTranslation, node, ~);
        nodeTranslation = translateDepthToSpace(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateDiv(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateDropout(~, nodeTranslation, node, ~);
        nodeTranslation = translateElu(~, nodeTranslation, node, ~);
        nodeTranslation = translateEqual(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateExp(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateFlatten(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateFloor(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateGather(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateGemm(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateGlobalAveragePool(~, nodeTranslation, node, ~);
        nodeTranslation = translateGlobalMaxPool(~, nodeTranslation, node, ~);
        nodeTranslation = translateGreater(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateGRU(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateHardmax(~, nodeTranslation, node, ~);
        nodeTranslation = translateHardSigmoid(~, nodeTranslation, node, ~);
        nodeTranslation = translateIdentity(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateIf(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateImageScaler(~, nodeTranslation, node, ~);
        nodeTranslation = translateInstanceNormalization(~, nodeTranslation, node, ~);
        nodeTranslation = translateLeakyRelu(~, nodeTranslation, node, ~);
        nodeTranslation = translateLess(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateLog(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateLogSoftmax(~, nodeTranslation, node, ~);
        nodeTranslation = translateLoop(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateLRN(~, nodeTranslation, node, ~);
        nodeTranslation = translateLSTM(~, nodeTranslation, node, ~);
        nodeTranslation = translateMatMul(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateMax(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateMaxPool(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateMin(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateMul(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateNeg(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateNot(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateOr(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translatePad(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translatePow(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translatePRelu(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateRandomUniform(~, nodeTranslation, node, ~);
        nodeTranslation = translateRandomUniformLike(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateReciprocal(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateReduceL1(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateReduceL2(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateReduceLogSum(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateReduceLogSumExp(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateReduceMax(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateReduceMean(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateReduceMin(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateReduceProd(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateReduceSum(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateReduceSumSquare(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateRelu(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateReshape(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateShape(~, nodeTranslation, node, ~);
        nodeTranslation = translateSigmoid(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateSize(~, nodeTranslation, node, ~);
        nodeTranslation = translateSoftmax(~, nodeTranslation, node, ~);
        nodeTranslation = translateSoftplus(~, nodeTranslation, node, ~);
        nodeTranslation = translateSlice(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateSpaceToDepth(~, nodeTranslation, node, ~);
        nodeTranslation = translateSplit(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateSqrt(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateSqueeze(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateSub(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateSum(~, nodeTranslation, node, ~);
        nodeTranslation = translateTanh(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateTile(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateTopK(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateTranspose(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateUnsqueeze(this, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateUnsupportedONNXLayers(~, nodeTranslation, node, ~);
        nodeTranslation = translateUnaryOperator(~, nodeTranslation, node, MATLABFcn, IntegerTensorNames);
        nodeTranslation = translateUnaryDlarrayOperator(~, nodeTranslation, node, MATLABFcn, IntegerTensorNames);
        nodeTranslation = translateUniBroadcastInfixBinaryOp(~, nodeTranslation, node, MATLABFcn, IntegerTensorNames);
        nodeTranslation = translateUniBroadcastPrefixBinaryOp(~, nodeTranslation, node, MATLABFcn, IntegerTensorNames);
        nodeTranslation = translateXor(this, nodeTranslation, node, ~);
    end
end