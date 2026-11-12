function nodeTranslation = translateAdd(this, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023 The MathWorks, Inc.

nodeTranslation = translateUniBroadcastInfixBinaryOp(this, nodeTranslation, node, '+', IntegerTensorNames);
end

