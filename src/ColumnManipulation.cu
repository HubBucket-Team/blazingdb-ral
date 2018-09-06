
//TODO: in theory  we want to get rid of this
// we should be using permutation iterators when we can

#include "ColumnManipulation.cuh"

#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/iterator/counting_iterator.h>

#include <thrust/execution_policy.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/transform_iterator.h>



template <typename InputType>
struct negative_to_zero : public thrust::unary_function< InputType, InputType>
{
	__host__ __device__
	InputType operator()(InputType x)
	{
		return x < 0 ? 0 : x;
	}
};


//TODO: this should probably be replaced with an api in libgdf that does the same, the
// following code exists in streamcompactionops.cu


//wow the freaking example from iterator_adaptpr, what a break right!
template<typename Iterator>
class repeat_iterator
		: public thrust::iterator_adaptor<
		  repeat_iterator<Iterator>, // the first template parameter is the name of the iterator we're creating
		  Iterator                   // the second template parameter is the name of the iterator we're adapting
		  // we can use the default for the additional template parameters
		  >
{
public:
	// shorthand for the name of the iterator_adaptor we're deriving from
	typedef thrust::iterator_adaptor<
			repeat_iterator<Iterator>,
			Iterator
			> super_t;
	__host__ __device__
	repeat_iterator(const Iterator &x, int n) : super_t(x), begin(x), n(n) {}
	// befriend thrust::iterator_core_access to allow it access to the private interface below
	friend class thrust::iterator_core_access;
private:
	// repeat each element of the adapted range n times
	unsigned int n;
	// used to keep track of where we began
	const Iterator begin;
	// it is private because only thrust::iterator_core_access needs access to it
	__host__ __device__
	typename super_t::reference dereference() const
	{
		return *(begin + (this->base() - begin) / n);
	}
};



typedef repeat_iterator<thrust::detail::normal_iterator<thrust::device_ptr<gdf_valid_type> > > gdf_valid_iterator;

size_t get_number_of_bytes_for_valid (size_t column_size) {
    return sizeof(gdf_valid_type) * (column_size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE;
}


// note: functor inherits from unary_function
struct modulus_bit_width : public thrust::unary_function<gdf_size_type,gdf_size_type>
{
	size_t n_bytes;
	size_t column_size;

	modulus_bit_width (size_t b_nytes, size_t column_size) {
		this->n_bytes = n_bytes;
		this->column_size = column_size;
	}
	__host__ __device__
	gdf_size_type operator()(gdf_size_type x) const
	{
		int col_position = x / 8;
        int length_col = n_bytes != col_position+1 ? GDF_VALID_BITSIZE : column_size - GDF_VALID_BITSIZE * (n_bytes - 1);
		//return x % GDF_VALID_BITSIZE;
		return (length_col - 1) - (x % 8);
		// x <<
	}
};


struct shift_left: public thrust::unary_function<gdf_valid_type,gdf_valid_type>
{

	gdf_valid_type num_bits;
	shift_left(gdf_valid_type num_bits): num_bits(num_bits){

	}

  __host__ __device__
  gdf_valid_type operator()(gdf_valid_type x) const
  {
    return x << num_bits;
  }
};

struct shift_right: public thrust::unary_function<gdf_valid_type,gdf_valid_type>
{

	gdf_valid_type num_bits;
	bool not_too_many;
	shift_right(gdf_valid_type num_bits, bool not_too_many)
		: num_bits(num_bits), not_too_many(not_too_many){

	}

  __host__ __device__
  gdf_valid_type operator()(gdf_valid_type x) const
  {
	    //if you want to force the shift to be fill bits with 0 you need to use an unsigned type

	  return *((unsigned char *) &x) >> num_bits;

  }
};

struct bit_or: public thrust::unary_function<thrust::tuple<gdf_valid_type,gdf_valid_type>,gdf_valid_type>
{


	__host__ __device__
	gdf_valid_type operator()(thrust::tuple<gdf_valid_type,gdf_valid_type> x) const
	{
		return thrust::get<0>(x) | thrust::get<1>(x);
	}
};


typedef thrust::transform_iterator<modulus_bit_width, thrust::counting_iterator<gdf_size_type> > bit_position_iterator;


template<typename stencil_type>
struct is_stencil_true
{
	__host__ __device__
	bool operator()(const thrust::tuple<stencil_type, gdf_valid_iterator::value_type, bit_position_iterator::value_type> value)
	{
		gdf_size_type position = thrust::get<2>(value);

		return ((thrust::get<1>(value) >> position) & 1) && (thrust::get<0>(value) != 0);
	}
};

struct is_bit_set
{
	__host__ __device__
	bool operator()(const thrust::tuple< gdf_valid_iterator::value_type, bit_position_iterator::value_type> value)
	{
		gdf_size_type position = thrust::get<1>(value);

		return ((thrust::get<0>(value) >> position) & 1);
	}
};

struct bit_mask_pack_op : public thrust::unary_function<int64_t,gdf_valid_type>
{
	__host__ __device__
		gdf_valid_type operator()(const int64_t expanded)
		{
			gdf_valid_type result = 0;
			for(int i = 0; i < GDF_VALID_BITSIZE; i++){
				// 0, 8, 16, ....,48,  56
				unsigned char byte = (expanded >> ( (GDF_VALID_BITSIZE - 1 - i )  * 8));
				result |= (byte & 1) << i;
			}
			return (result);
		}
};


typedef struct packed_ints {
	int _0;
	int _1;
	int _2;
	int _3;
	int _4;
	int _5;
	int _6;
	int _7;
} packed_ints;


//struct reorder_bitmask : public thrust::unary_function<packed_ints,

/*
template <typename ElementIterator, typename IndexIterator, typename OutputIterator>
gdf_error materialize_templated_3(gdf_column * input, gdf_column * output, gdf_column * row_indeces){

 * TODO: materialize bitmask
	//we can use the output gdf as a space to process the bitmasks in
	typedef thrust::tuple<gdf_valid_iterator, bit_position_iterator > mask_tuple;
		typedef thrust::zip_iterator<mask_tuple> zipped_mask;


		zipped_mask  zipped_mask_iter(
				thrust::make_tuple(
						valid_iterator,
						thrust::make_transform_iterator<modulus_bit_width, thrust::counting_iterator<gdf_size_type> >(
								thrust::make_counting_iterator<gdf_size_type>(0),
								modulus_bit_width(n_bytes, stencil->size))
				)
		);

		typedef thrust::transform_iterator<is_bit_set, zipped_mask > bit_set_iterator;
		bit_set_iterator bit_set_iter = thrust::make_transform_iterator<is_bit_set,zipped_mask>(
				zipped_mask_iter,
				is_bit_set()
		);



	//any negative values could fuck this up so i gues we need a transformation to filter them out

	//TODO: right now we expand this out to fucking 1 byte wide, gross

	thrust::detail::normal_iterator<thrust::device_ptr<ElementIterator> > element_iter =
			thrust::detail::make_normal_iterator(thrust::device_pointer_cast((ElementIterator *) input->data));

	thrust::transform_iterator<thrust::detail::normal_iterator<thrust::device_ptr<ElementIterator> >,thrust::detail::normal_iterator<thrust::device_ptr<ElementIterator> >> transform_iter =
			thrust::make_transform_iterator(element_iter,negative_to_zero());

	thrust::detail::normal_iterator<thrust::device_ptr<IndexIterator> > index_iter =
			thrust::detail::make_normal_iterator(thrust::device_pointer_cast((IndexIterator *) row_indeces->data));
	thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<ElementIterator> >,thrust::detail::normal_iterator<thrust::device_ptr<IndexIterator> >> iter(transform_iter,index_iter);

	thrust::detail::normal_iterator<thrust::device_ptr<IndexIterator> > output_iter =
			thrust::detail::make_normal_iterator(thrust::device_pointer_cast((OutputIterator *) output->data));;
	thrust::copy(iter,iter + input->size,output_iter);

	return GDF_SUCCESS;

}*/

//input and output shoudl be the same time
template <typename ElementIterator, typename IndexIterator>
gdf_error materialize_templated_2(gdf_column * input, gdf_column * output, gdf_column * row_indeces){
	//TODO: handle the bitmasks

	thrust::detail::normal_iterator<thrust::device_ptr<ElementIterator> > element_iter =
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast((ElementIterator *) input->data));

	thrust::detail::normal_iterator<thrust::device_ptr<IndexIterator> > index_iter =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((IndexIterator *) row_indeces->data));

	typedef thrust::detail::normal_iterator<thrust::device_ptr<IndexIterator> > IndexNormalIterator;

	thrust::transform_iterator<negative_to_zero<IndexIterator>,IndexNormalIterator> transform_iter = thrust::make_transform_iterator(index_iter,negative_to_zero<IndexIterator>());


		thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<ElementIterator> >,thrust::transform_iterator<negative_to_zero<IndexIterator>,IndexNormalIterator> > iter(element_iter,transform_iter);

		thrust::detail::normal_iterator<thrust::device_ptr<ElementIterator> > output_iter =
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast((ElementIterator *) output->data));;
		thrust::copy(iter,iter + input->size,output_iter);

		return GDF_SUCCESS;
}

template <typename ElementIterator>
gdf_error materialize_templated_1(gdf_column * input, gdf_column * output, gdf_column * row_indeces){
	int column_width;
	get_column_byte_width(row_indeces, &column_width);
	if(column_width == 1){
		return materialize_templated_2<ElementIterator,int8_t>(input,output,row_indeces);
	}else if(column_width == 2){
		return materialize_templated_2<ElementIterator,int16_t>(input,output,row_indeces);
	}else if(column_width == 4){
		return materialize_templated_2<ElementIterator,int32_t>(input,output,row_indeces);
	}else if(column_width == 8){
		return materialize_templated_2<ElementIterator,int64_t>(input,output,row_indeces);
	}

}


gdf_error materialize_column(gdf_column * input, gdf_column * output, gdf_column * row_indeces){
	int column_width;
	get_column_byte_width(input, &column_width);
	if(column_width == 1){
		return materialize_templated_1<int8_t>(input,output,row_indeces);
	}else if(column_width == 2){
		return materialize_templated_1<int16_t>(input,output,row_indeces);
	}else if(column_width == 4){
		return materialize_templated_1<int32_t>(input,output,row_indeces);
	}else if(column_width == 8){
		return materialize_templated_1<int64_t>(input,output,row_indeces);
	}


}

