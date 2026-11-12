function nodeTranslation = translateReduceSumSquare(this, nodeTranslation, node, IntegerTensorNames)
    %   Copyright 2024 The MathWorks, Inc. 

    if ismember(node.input{1}, IntegerTensorNames)
        op = '%s = sum(fix(%s.^2), %s);\n';
    else
        op = '%s = sum(%s.^2, %s);\n';
    end
    nodeTranslation = translateReduceOp(this, nodeTranslation, node, IntegerTensorNames, op);
end
