function nodeTranslation = translateLess(this, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024 The MathWorks, Inc.

nodeTranslation = translateInfixBinaryOperator(this, nodeTranslation, node, '<', IntegerTensorNames);
end

