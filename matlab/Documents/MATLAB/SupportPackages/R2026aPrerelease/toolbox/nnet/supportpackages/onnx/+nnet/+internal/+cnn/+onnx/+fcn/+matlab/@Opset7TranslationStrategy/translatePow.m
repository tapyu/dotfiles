function nodeTranslation = translatePow(this, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023 The MathWorks, Inc.

nodeTranslation = translatePrefixBinaryOperator(this, nodeTranslation, node, 'power', IntegerTensorNames);
end

