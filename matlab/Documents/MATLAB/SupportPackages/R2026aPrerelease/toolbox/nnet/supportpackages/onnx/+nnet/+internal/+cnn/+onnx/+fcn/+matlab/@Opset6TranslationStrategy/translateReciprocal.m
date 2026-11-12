function nodeTranslation = translateReciprocal(this, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023 The MathWorks, Inc.

nodeTranslation = translateUnaryOperator(this, nodeTranslation, node, '1./', IntegerTensorNames);
end

