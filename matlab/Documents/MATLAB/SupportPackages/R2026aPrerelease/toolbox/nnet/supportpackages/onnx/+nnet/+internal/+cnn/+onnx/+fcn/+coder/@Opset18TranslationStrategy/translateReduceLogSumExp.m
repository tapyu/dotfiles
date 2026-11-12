function nodeTranslation = translateReduceLogSumExp(this, nodeTranslation, node, IntegerTensorNames)
    %   Copyright 2024 The MathWorks, Inc. 

    if ismember(node.input{1}, IntegerTensorNames)
        op = '%s = fix(log(sum(fix(exp(%s)), %s)));\n';
    else
        op = '%s = log(sum(exp(%s), %s));\n';
    end
    nodeTranslation = translateReduceOp(this, nodeTranslation, node, IntegerTensorNames, op);
end