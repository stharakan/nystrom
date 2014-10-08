function Aout = scale_to_one(A)
  [n,dim]=size(A);
  amin = min(A);
  amax = max(A);
  da = amax-amin;
  da = 1./da;
  nanidx = find(isinf(da)); da(nanidx)=1;   % correct for dimensions that have no variability. 
  Aout = A-repmat(amin,n,1);
  Aout = Aout.*repmat(da,n,1);
  
  
