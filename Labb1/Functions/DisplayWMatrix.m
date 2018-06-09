function DisplayWMatrix(W)

    for i=1:10
        im = reshape(W(i, :), 32, 32, 3);
        s_im{i} = (im - min(im(:)))  / (max(im(:))  - min(im(:)));
        s_im{i}= permute(s_im{i}, [2, 1, 3]);
    end
    imshow(W);
end