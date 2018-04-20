/*******************************************************************************/
/* File     config.hpp
/* Author   Johannes Braun
/* Created  31.03.2018
/*
/* User space for own settings.
/*******************************************************************************/

#pragma once

// CUSTOM LOGGING:
// If you want to use your own logger for error logs, you can define ERR_OUTPUT(x) with x being the logged string.
// Example for a custom stream-style logger:
// #define ERR_OUTPUT(x) my_logger("GLShader") << (x)

// NAMESPACE:
namespace glshader::process {}
// Shorten base namespace. You can use your own namespace if you wish.
namespace glsp = glshader::process;