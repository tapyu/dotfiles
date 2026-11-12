function nodeTranslation = translateReciprocal(this, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024 The MathWorks, Inc.

nodeTranslation = translateUnaryOperator(this, nodeTranslation, node, '1./', IntegerTensorNames);
end

