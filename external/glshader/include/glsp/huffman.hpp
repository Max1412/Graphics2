/*******************************************************************************/
/* File     huffman.hpp
/* Author   Johannes Braun
/* Created  31.03.2018
/*
/* Provides helper functionality for compressing and uncompressing data using
/* huffman trees.
/*******************************************************************************/

#pragma once

#include "config.hpp"
#include <array>
#include <vector>
#include <queue>
#include <sstream>

namespace glshader::process::compress::huffman
{
    /* Tests a type on whether it is a container type by checking for a value_type type, as well as a resize(size_t) function and 
    overloads for std::size(...) and std::data(...). */
    template<typename Container, typename BaseContainer = std::decay_t<std::remove_const_t<Container>>> 
    using enable_if_container = std::void_t<
        typename BaseContainer::value_type,
        decltype(std::declval<BaseContainer>().resize(size_t(0))),
        decltype(std::size(std::declval<BaseContainer>())),
        decltype(std::data(std::declval<BaseContainer>()))
    >;

    /* Contains a byte stream used when de-/encoding.
    For simple std::basic_string<uint8_t> conversion, call stream.stringstream.str(), otherwise you can convert it 
    to any other STL contiguous-storage container using to_container<Container>(). */
    struct stream {
        size_t stream_length;
        std::basic_stringstream<uint8_t> stringstream;

        template<typename Container, typename = enable_if_container<Container>>
        std::decay_t<std::remove_const_t<Container>> to_container()
        {
            using BaseContainer = std::decay_t<std::remove_const_t<Container>>;
            BaseContainer container;
            container.resize(stream_length / sizeof(typename BaseContainer::value_type));
            stringstream.read(reinterpret_cast<uint8_t*>(std::data(container)), std::size(container) * sizeof(typename BaseContainer::value_type));
            return container;
        }
    };

    /*******************************/
    /*  STL container wrapper
    /*******************************/

    /* Helper function calling encode(const uint8_t*, size_t) */
    template<typename Container, typename = enable_if_container<Container>>
    stream encode(const Container& in) { return encode(std::data(in), std::size(in)); }

    /* Helper function calling decode(const uint8_t*, size_t) */
    template<typename Container, typename = enable_if_container<Container>>
    stream decode(const Container& in) { return decode(std::data(in), std::size(in)); }

    /*******************************/
    /*  Base functions
    /*******************************/

    /* Encode a given uncompressed input with a given length into a compressed stream form using the huffman algorithm. */
    stream encode(const uint8_t* in, size_t in_length);

    /* Encode a given compressed input with a given length into an uncompressed stream form using the huffman algorithm. */
    stream decode(const uint8_t* in, size_t in_length);
}