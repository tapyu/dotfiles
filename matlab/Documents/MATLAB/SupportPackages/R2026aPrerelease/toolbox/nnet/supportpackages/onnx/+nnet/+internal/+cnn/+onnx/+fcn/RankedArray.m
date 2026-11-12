classdef RankedArray
    % An array with a fixed rank. 1D arrays are automatically converted to
    % column vectors.
    
    %   Copyright 2020 The MathWorks, Inc.    

    properties
        Array
        Rank
    end
    
    methods
        function obj = RankedArray(array, rnk)
            if rnk==1
                obj.Array = array(:);
            else
                obj.Array = array;
            end
            obj.Rank = rnk;
        end
    end
end

