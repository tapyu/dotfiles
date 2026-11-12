function nodeTranslation = translateGreater(this, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023 The MathWorks, Inc.

nodeTranslation = translateInfixBinaryOperator(this, nodeTranslation, node, '>', IntegerTensorNames);
end

