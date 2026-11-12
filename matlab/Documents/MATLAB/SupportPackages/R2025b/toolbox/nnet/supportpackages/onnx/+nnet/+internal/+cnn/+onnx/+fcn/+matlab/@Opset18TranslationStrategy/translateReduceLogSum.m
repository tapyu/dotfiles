function nodeTranslation = translateReduceLogSum(this, nodeTranslation, node, IntegerTensorNames)
    %   Copyright 2024 The MathWorks, Inc. 

    if ismember(node.input{1}, IntegerTensorNames)
        op = '%s = fix(log(sum(%s, dims)));\n';
    else
        op = '%s = log(sum(%s, dims));\n';
    end
    nodeTranslation = translateReduceOp(this, nodeTranslation, node, IntegerTensorNames, op);
end