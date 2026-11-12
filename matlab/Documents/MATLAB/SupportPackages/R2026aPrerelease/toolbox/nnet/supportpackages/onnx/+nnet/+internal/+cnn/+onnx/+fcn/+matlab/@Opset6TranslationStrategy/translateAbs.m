function nodeTranslation = translateAbs(this, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023 The MathWorks, Inc.

nodeTranslation = translateUnaryOperator(this, nodeTranslation, node, 'abs', IntegerTensorNames);
end

