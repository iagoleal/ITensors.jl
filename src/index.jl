
const IDType = UInt64

# Arrow direction
@enum Arrow In=-1 Out=1 Neither=0

function -(dir::Arrow)
  if dir==Neither
    error("Cannot reverse direction of Arrow direction 'Neither'")
  else
    return dir==In ? Out : In
  end
end

struct Index
  id::IDType
  dim::Int
  dir::Arrow
  plev::Int
  tags::TagSet
  Index(id::IDType,
        dim::Integer,
        dir::Arrow,
        plev::Integer,
        tags::TagSet) = new(id,dim,dir,plev,tags)
end

Index() = Index(IDType(0),1,Neither,0,TagSet(""))
Index(dim::Integer,tags::String="") = Index(rand(IDType),dim,In,0,TagSet(tags))

id(i::Index) = i.id
dim(i::Index) = i.dim
function dim(i1::Index,inds::Index...)
  total_dim = 1
  total_dim *= dim(i1)*dim(inds...)
  return total_dim
end
dim() = 1
dir(i::Index) = i.dir
plev(i::Index) = i.plev
tags(i::Index) = i.tags

==(i1::Index,i2::Index) = (id(i1)==id(i2) && plev(i1)==plev(i2) && tags(i1)==tags(i2))
copy(i::Index) = Index(i.id,i.dim,i.dir,i.plev,copy(i.tags))

dag(i::Index) = Index(id(i),dim(i),-dir(i),plev(i),tags(i))

isdefault(i::Index) = (i==Index())

function setprime(i::Index,plev::Int)
  return Index(id(i),dim(i),dir(i),plev,tags(i))
end
noprime(i::Index) = setprime(i,0)

addtags(i::Index,ts::AbstractString) = Index(id(i),dim(i),dir(i),plev(i),addtags(tags(i),TagSet(ts)))
removetags(i::Index,ts::AbstractString) = Index(id(i),dim(i),dir(i),plev(i),removetags(tags(i),TagSet(ts)))
settags(i::Index,ts::AbstractString) = Index(id(i),dim(i),dir(i),plev(i),TagSet(ts))
hastags(i::Index,ts::Union{AbstractString,TagSet}) = hastags(tags(i),ts)

#prime!(i::Index,plinc::Int=1) = (i.plev+=plinc; return i)
function prime(i::Index,inc::Int=1)
  return Index(id(i),dim(i),dir(i),plev(i)+inc,tags(i))
end
adjoint(i::Index) = prime(i)

#(i::Index)(tags::String) = settags(i,tags)
function replacetags(i::Index,tsold::AbstractString,tsnew::AbstractString) 
  tagsetold = TagSet(tsold)
  #TODO: Avoid this copy?
  tagsetold∉tags(i) && return copy(i)
  itags = addtags(removetags(tags(i),tagsetold),TagSet(tsnew))
  return Index(id(i),dim(i),dir(i),plev(i),itags)
end

function tags(i::Index,ts::AbstractString)
  ts = filter(x -> !isspace(x),ts)
  vts = split(ts,"->")
  length(vts) == 1 && error("Must use -> to replace tags of an Index")
  length(vts) > 2 && error("Can only use a single -> when replacing tags of an Index")
  tsremove,tsadd = vts
  if tsremove==""
    return addtags(i,tsadd)
  #TODO: notation to replace all tags?
  #elseif tsremove=="all"
  #  ires = settags(i,tsadd)
  elseif tsadd==""
    return removetags(i,tsremove)
  else
    return replacetags(i,tsremove,tsadd)
  end
end

# Iterating over Index I gives
# integers from 1...dim(I)
start(i::Index) = 1
next(i::Index,n::Int) = (n,n+1)
done(i::Index,n::Int) = (n > dim(i))
colon(n::Int,i::Index) = range(n,dim(i))

function primeString(i::Index)
  pl = plev(i)
  if pl == 0 return ""
  elseif pl > 3 return "'$pl"
  else return "'"^pl
  end
end

function show(io::IO,i::Index) 
    idstr = "$(id(i) % 1000)"
    if length(tags(i)) > 0
      print(io,"($(dim(i)),$(tags(i))|id=$(idstr))$(primeString(i))")
    else
      print(io,"($(dim(i))|id=$(idstr))$(primeString(i))")
    end
end

struct IndexVal
  ind::Index
  val::Int
  function IndexVal(i::Index,n::Int)
    n>dim(i) && throw(ErrorException("Value $n greater than size of Index $i"))
    n<1 && throw(ErrorException("Index value must be >= 1 (was $n)"))
    return new(i,n)
  end
end
(i::Index)(n::Int) = IndexVal(i,n)

val(iv::IndexVal) = iv.val
ind(iv::IndexVal) = iv.ind

==(i::Index,iv::IndexVal) = (i==ind(iv))
==(iv::IndexVal,i::Index) = (i==iv)

plev(iv::IndexVal) = plev(ind(iv))
prime(iv::IndexVal,inc::Integer=1) = IndexVal(prime(ind(iv),inc),val(iv))
adjoint(iv::IndexVal) = IndexVal(adjoint(ind(iv)),val(iv))

show(io::IO,iv::IndexVal) = print(io,ind(iv),"=$(val(iv))")
