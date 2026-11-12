function nodeTranslation = translateCeil(this, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023 The MathWorks, Inc.

nodeTranslation = translateUnaryOperator(this, nodeTranslation, node, 'ceil', IntegerTensorNames);
end

