Messy tables
from messytables import CSVTableSet, XLSTableSet, PDFTableSet, type_guess, types_processor, headers_guess, \
    headers_processor, offset_processor, any_tableset

suite = [{'filename': 'gen_input_data.csv', 'tableset': CSVTableSet},
         {'filename': 'gen_input_data.xls', 'tableset': XLSTableSet},
         {'filename': 'gen_input_data.xlsx', 'tableset': XLSTableSet},
         {'filename': 'gen_input_data.pdf', 'tableset': PDFTableSet}
         ]

# Special handling for PDFTables - skip if not installed
try:
    import pdftables
except ImportError:
    got_pdftables = False
    suite.append({"filename": "gen_input_data.pdf", "tableset": False})
else:
    from messytables import PDFTableSet
    got_pdftables = True
    suite.append({"filename": "gen_input_data.pdf", "tableset": PDFTableSet})


# Open the file for read
    #input_file = open(input_path, 'rb')

    # Load the input file and assert file type
    #input_table_set = any_tableset(input_file)

    #input_file_ext = filter(lambda d:d['filename']== input_filename, suite)

    #raises assertion error (need to check)
    #assert isinstance(input_table_set, input_file_ext[0]['tableset']), type(input_table_set)

    #fecthing the data
    #row_set = input_table_set.tables[0]
    #data = list(row_set)

    #sampling the data - only run once
    #print row_set.sample.next()

    #length of data
    #print len(data)

    # guess header names and the offset of the header: (only works when headers are present
    #  if headers are not present it wrongly takes the first row as header)
    #offset, headers = headers_guess(row_set.sample)
    #row_set.register_processor(headers_processor(headers))
    #print offset

    #Print the column names for data set
    #if headers:
    #    for clmn in headers:
    #        print clmn

    # add one to begin with content, not the header and print next line (file with header) - just offsets the read
    #row_set.register_processor(offset_processor(offset + 1))

    #column type value of every cell says string?
#    for row in row_set:
#        for clmn in row:
#            print clmn.value , " ", clmn.type

    # guess column types: Guess column based on header - makes it all strings - header needs to be clipped off
    # presence of blanks and nulls also makes it string
    #types = type_guess(row_set.sample, strict=True)
    #print types

    # and tell the row set to apply these types to
    # each row when traversing the iterator: sanity check of data
    #row_set.register_processor(types_processor(types))

    # now run some operation on the data:
    #for row in row_set:
    #    do_something(row)

=============
