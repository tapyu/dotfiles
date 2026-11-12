function str = sprintfVector(vec)

%   Copyright 2020 The MathWorks, Inc.    

str = sprintf('[');
str = [str sprintf('%d ', vec)];
str = [str sprintf(']')];
end