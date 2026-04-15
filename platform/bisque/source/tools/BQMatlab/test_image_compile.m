imurl = 'https://bisque.example.com/data_service/image/161855';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% using bq.Image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fetch image into a file using its original name
image = bq.Factory.fetch(imurl);
filename = image.fetch([]);

