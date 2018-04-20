#include "eval.hpp"

#include "control.hpp"
#include "../strings.hpp"

#include <cstring>
#include <list>

namespace glshader::process::impl::operation
{
    enum eval_op
    {
        /*** http://de.cppreference.com/w/cpp/language/operator_precedence ***/
        /*** priority: 3 ***/
        op_neg = 0, op_pos = 1,
        op_not = 2, op_inv = 3,

        /*** priority: 5 ***/
        op_mul = 4, op_div = 5,
        op_mod = 6,

        /*** priority: 6 ***/
        op_add = 7, op_sub = 8,

        /*** priority: 8 ***/
        op_lt = 9, op_leq = 10,
        op_gt = 11, op_geq = 12,

        /*** priority: 9 ***/
        op_eq = 13, op_neq = 14,

        /*** priority: 10 ***/
        op_and = 15,

        /*** priority: 11 ***/
        op_xor = 16,

        /*** priority: 12 ***/
        op_or = 17,

        /*** priority: 13 ***/
        op_land = 18,

        /*** priority: 14 ***/
        op_lor = 19,
    };

    struct eval_item
    {
        const char* first_after_op;
        const char* first_of_op;
        eval_op o;
    };
    
    bool substreq(const char* x, const char* y, int len)
    {
        return std::strncmp(x, y, len) == 0;
    }

    int eval_operator(const char* begin, const eval_item& o, int len, const files::path& file, const int line);
    int eval(const char* x, int len, const files::path& file, const int line)
    {
        enum state
        {
            def = 0,
            inop = 1,
        };

        state s = def;
        std::list<eval_item> opstack;

        while (*x == ' ' || *x == '\t')
        {
            ++x;
            --len;
        }
        while (*(x + len - 1) == ' ' || *(x + len - 1) == '\t')
            --len;

        if (*x == '(' && *(x + len - 1) == ')')
            ++x, len -= 2;

        const char* begin = x;

        for (auto c = x; c < x + len; ++c)
        {
            if (s == def)
            {
                if (substreq(c, "-", 1))      opstack.push_back({ c + 1, c, op_neg });
                else if (substreq(c, "+", 1)) opstack.push_back({ c + 1, c, op_pos });
                else if (substreq(c, "!", 1)) opstack.push_back({ c + 1, c, op_not });
                else if (substreq(c, "~", 1)) opstack.push_back({ c + 1, c, op_inv });
                else if (substreq(c, "(", 1))
                {
                    for (int stk = 1; stk != 0; )
                    {
                        ++c;
                        if (c - begin > len)
                        {
                            syntax_error(file, line, strings::serr_eval_end_of_brackets);
                            return 0;
                        }
                        if (*c == '(') ++stk;
                        if (*c == ')') --stk;
                    }
                    ++c;
                    s = inop;
                }
            }
            else if (s == inop)
            {
                if      (substreq(c, "*",  1)) opstack.push_back({ c + 1, c, op_mul  });
                else if (substreq(c, "/",  1)) opstack.push_back({ c + 1, c, op_div  });
                else if (substreq(c, "%",  1)) opstack.push_back({ c + 1, c, op_mod  });
                else if (substreq(c, "+",  1)) opstack.push_back({ c + 1, c, op_add  });
                else if (substreq(c, "-",  1)) opstack.push_back({ c + 1, c, op_sub  });
                else if (substreq(c, "<=", 2)) opstack.push_back({ c + 2, c, op_leq  });
                else if (substreq(c, "<",  1)) opstack.push_back({ c + 1, c, op_lt   });
                else if (substreq(c, ">=", 2)) opstack.push_back({ c + 2, c, op_geq  });
                else if (substreq(c, ">",  1)) opstack.push_back({ c + 1, c, op_gt   });
                else if (substreq(c, "==", 2)) opstack.push_back({ c + 2, c, op_eq   });
                else if (substreq(c, "!=", 2)) opstack.push_back({ c + 2, c, op_neq  });
                else if (substreq(c, "&&", 2)) opstack.push_back({ c + 2, c, op_land });
                else if (substreq(c, "||", 2)) opstack.push_back({ c + 2, c, op_lor  });
                else if (substreq(c, "&",  1)) opstack.push_back({ c + 1, c, op_and  });
                else if (substreq(c, "^",  1)) opstack.push_back({ c + 1, c, op_xor  });
                else if (substreq(c, "|",  1)) opstack.push_back({ c + 1, c, op_or   });
            }

            if (isdigit(*c))
                s = inop;
            else if (*c != ' ')
                s = def;
        }

        if (opstack.empty())
            return std::stoi(std::string(begin, begin + len));

        eval_op ox = opstack.front().o;
        auto limit = opstack.begin();
        for (auto it = opstack.begin(); it != opstack.end(); ++it)
            if (it->o > ox)
            {
                ox = it->o;
                limit = it;
            }

        if (limit == opstack.end())
            limit = opstack.begin();

        return eval_operator(begin, *limit, len, file, line);
    }

    int eval_operator(const char* begin, const eval_item& o, int len, const files::path& file, const int line)
    {
        const int num_begin = int(o.first_of_op - begin);
        const int num_end = int(len - (o.first_after_op - begin));

        /*** Curious switch :P ***/
        switch (o.o)
        {
        case op_neg:    return - eval(o.first_after_op, num_end, file, line);
        case op_pos:    return + eval(o.first_after_op, num_end, file, line);
        case op_not:    return ! eval(o.first_after_op, num_end, file, line);
        case op_inv:    return ~ eval(o.first_after_op, num_end, file, line);
        case op_mul:    return eval(begin, num_begin, file, line) *  eval(o.first_after_op, num_end, file, line);
        case op_div:    return eval(begin, num_begin, file, line) /  eval(o.first_after_op, num_end, file, line);
        case op_mod:    return eval(begin, num_begin, file, line) %  eval(o.first_after_op, num_end, file, line);
        case op_add:    return eval(begin, num_begin, file, line) +  eval(o.first_after_op, num_end, file, line);
        case op_sub:    return eval(begin, num_begin, file, line) -  eval(o.first_after_op, num_end, file, line);
        case op_lt:     return eval(begin, num_begin, file, line) <  eval(o.first_after_op, num_end, file, line);
        case op_leq:    return eval(begin, num_begin, file, line) <= eval(o.first_after_op, num_end, file, line);
        case op_gt:     return eval(begin, num_begin, file, line) >  eval(o.first_after_op, num_end, file, line);
        case op_geq:    return eval(begin, num_begin, file, line) >= eval(o.first_after_op, num_end, file, line);
        case op_eq:     return eval(begin, num_begin, file, line) == eval(o.first_after_op, num_end, file, line);
        case op_neq:    return eval(begin, num_begin, file, line) != eval(o.first_after_op, num_end, file, line);
        case op_and:    return eval(begin, num_begin, file, line) &  eval(o.first_after_op, num_end, file, line);
        case op_xor:    return eval(begin, num_begin, file, line) ^  eval(o.first_after_op, num_end, file, line);
        case op_or:     return eval(begin, num_begin, file, line) |  eval(o.first_after_op, num_end, file, line);
        case op_land:   return eval(begin, num_begin, file, line) && eval(o.first_after_op, num_end, file, line);
        case op_lor:    return eval(begin, num_begin, file, line) || eval(o.first_after_op, num_end, file, line);
        default:        return 0;
        }
    }
}