#ifndef __H5SX_HPP__
#define __H5SX_HPP__
#include <H5Cpp.h>

class H5DataSetBase {
protected:
    H5::DataSet dset;
    H5::DataType dtype;
public:
    std::string name;
    virtual void write(const void* data, const hsize_t start[], const hsize_t count[]) = 0;
};

template<size_t RANK>
class H5DataSet : public H5DataSetBase {
public:
    void init(const H5::H5File& file, const std::string name, const H5::DataType dtype, const hsize_t dims[RANK]) {
	this->name = name;
	this->dtype = dtype;
	dset = file.createDataSet(name, dtype, H5::DataSpace(RANK, dims));
    }
    virtual void write(const void* data, const hsize_t start[RANK], const hsize_t count[RANK]) {
	H5::DataSpace dspace = dset.getSpace();
	dspace.selectHyperslab(H5S_SELECT_SET, count, start);
	dset.write(data, dtype, H5::DataSpace(RANK, count), dspace);
    }
};

template<size_t RANK>
class H5ExtendableDataSet : public H5DataSetBase {
public:
    void init(const H5::H5File& file, const std::string name, const H5::DataType dtype, const hsize_t dims[RANK],
	      const hsize_t max_dims[RANK], const hsize_t chunk_dims[RANK]) {
	this->name = name;
	this->dtype = dtype;
	H5::DSetCreatPropList cparms;
	cparms.setChunk(RANK, chunk_dims);
	dset = file.createDataSet(name, dtype, H5::DataSpace(RANK, dims, max_dims), cparms);
    }
    virtual void write(const void* data, const hsize_t start[RANK], const hsize_t count[RANK]) {
	hsize_t size[RANK];
	for(size_t i = 0; i < RANK; i++)
	    size[i] = start[i] + count[i];

	dset.extend(size);

	H5::DataSpace dspace = dset.getSpace();
	dspace.selectHyperslab(H5S_SELECT_SET, count, start);
	dset.write(data, dtype, H5::DataSpace(RANK, count), dspace);
    }
};

class H5File {
private:
    H5::H5File file;
    std::vector<H5DataSetBase*> data_sets;
public:
    void init(const std::string path, bool overwrite) {
	file = H5::H5File(path, overwrite?H5F_ACC_TRUNC:H5F_ACC_EXCL);
    }

    template<size_t RANK>
    void add_data_set(const std::string name, const H5::DataType dtype, const hsize_t dims[RANK]) {
	H5DataSet<RANK>* ds = new H5DataSet<RANK>();;
	ds->init(file, name, dtype, dims);
	data_sets.push_back(ds);
    }

    template<size_t RANK>
    void add_extendable_data_set(const std::string name, const H5::DataType dtype, const hsize_t dims[RANK],
				 const hsize_t max_dims[RANK], const hsize_t chunk_size[RANK]) {
	H5ExtendableDataSet<RANK>* ds = new H5ExtendableDataSet<RANK>();;
	ds->init(file, name, dtype, dims, max_dims, chunk_size);
	data_sets.push_back(ds);
    }

    template<size_t RANK>
    void write(const std::string ds_name, const void* data, const hsize_t start[RANK], const hsize_t count[RANK]) {
	for(H5DataSetBase* ds : data_sets) {
	    if(ds->name == ds_name) {
		ds->write(data, start, count);
		return;
	    }
	}
    }
    void flush() {
	if (file.getId() > 0) {
            file.flush(H5F_SCOPE_GLOBAL);
	}
    }
};

#endif
