function newimg = mycrop (img, padsize, padval)

[nrows, ncols, cols]=size(img);

if padsize(1)<nrows || padsize(2)<ncols
    error('ERROR: image is larger than target size.')
end

Dy = fix((padsize(1)-nrows)/2);
Dx = fix((padsize(2)-ncols)/2);


newimg = [repmat(padval, [padsize(1) Dx cols]) ...
    [repmat(padval, [Dy ncols cols]); img; repmat(padval, [padsize(1)-nrows-Dy ncols cols])] ...
    repmat(padval, [padsize(1) padsize(2)-ncols-Dx cols])];

end