function nodeTranslation = translateSqrt(this, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024 The MathWorks, Inc.

nodeTranslation = translateUnaryOperator(this, nodeTranslation, node, 'sqrt', IntegerTensorNames);
end

