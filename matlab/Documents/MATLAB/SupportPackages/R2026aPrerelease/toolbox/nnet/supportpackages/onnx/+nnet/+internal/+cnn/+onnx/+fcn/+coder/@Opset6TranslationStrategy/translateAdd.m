function nodeTranslation = translateAdd(this, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024 The MathWorks, Inc.

nodeTranslation = translateUniBroadcastInfixBinaryOp(this, nodeTranslation, node, '+', IntegerTensorNames);
end

