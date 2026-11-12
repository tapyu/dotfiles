function X = makeUnlabeledDlarray(X)
% Make numeric X into an unlabelled dlarray

%   Copyright 2020-2021 The MathWorks, Inc.    

if isa(X, 'dlarray')
    X = stripdims(X);
elseif isnumeric(X)
    if isinteger(X)
        % Make ints double so they can combine with anything without
        % reducing precision
        X = double(X);
    end
    X = dlarray(X);
end
end
