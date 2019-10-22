#Creates new data dictionary from dataset and replaces existing with new one
@analytics_toolsets.route('/featureStatistics',methods=['PUT'])
class generateFeatureStats(Resource):
    @api.response(*response_400)
    @api.expect(BaseSchema.data_dictionary_post)
    def put(self, model_meta_id, data_dictionary_version_id):
        """Creates a new data dictionary from a dataset or replaces the existing data dictionary with a given one"""
        data = request.json

        try:
            data_dict = DataDict(os_credentials=objectstore_credentials, os_container=data['os_container'],
            object_name=data['object_name'], object_version=data_dictionary_version_id)
            if data['input_format'] == 'file':
                dd = data_dict.generate_and_store(inp_form=data['input_format'], input_path=data['input_path'])
            else:
                dd = data_dict.generate_and_store(inp_form=data['input_format'], db_url=data['db_url'],
                    schema=data['schema'], tables=data['tables'], observation_ids=data['oids'])
        except Exception as e:
            raise exceptions.ParseError("Unable to generate Feature Statistics for this input")

        return 201

