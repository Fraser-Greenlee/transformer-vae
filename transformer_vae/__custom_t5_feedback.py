'''
    I was going to add "feedback" to the transforrmer but never finished it.

    Hard to know if it will help a T5 decoder with >=12 layers.

    Idea is to simply process the sequence over several "feedback windows".

    The final hidden states are then used at every level when processing the next window & so on.

    Ideally this will give the model more representational power while using less memory & more training time.
'''

"""
import math
import torch
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


logger = logging.get_logger(__name__)


def t5_block_alt(self, config, layer_id):
    '''
        Change methods in existing T5Block instance.
        Helps getting around init issues.
    '''
    self.layer_id = layer_id
    self.d_model = config.t5.d_model
    self.true_forward = self.forward

    def alt_forward(
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        encoder_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        n_feedback_tokens=0,
    ):
        outputs = self.true_forward(
            hidden_states,
            attention_mask,
            position_bias,
            encoder_hidden_states,
            encoder_attention_mask,
            encoder_decoder_position_bias,
            layer_head_mask,
            encoder_layer_head_mask,
            past_key_value,
            use_cache,
            output_attentions,
            return_dict,
        )

        if self.training:
            if n_feedback_tokens:
                # use the old feedback tokens
                batch_size = hidden_states.size(0)
                new_hidden_states = outputs[0].reshape(batch_size, -1, self.d_model)
                new_hidden_states[:, :n_feedback_tokens] = hidden_states[:, :n_feedback_tokens]
                import pdb;
                pdb.set_trace()
                return (new_hidden_states,) + outputs[1:]

        return outputs

    self.forward = alt_forward
    return self


def modify_t5_stack(self, config):
    self.window_mode = False
    if config.attention_window_size:
        for i, v in enumerate(self.block):
            self.block[i] = t5_block_alt(v, config, i)

    self.feedback_window_size = 0
    self.feedback_window_size = config.feedback_window_size
    for i, v in enumerate(self.block):
        self.block[i] = t5_block_alt(v, config, i)

    def alt_forward_feedback(
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        grad_chk_pnt_rate=None
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if self.training and use_cache:
            assert(grad_chk_pnt_rate is None), "Can't use grad checkpoint and cache."
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        ### CHANGE BELOW

        hidden_states = self.dropout(inputs_embeds)
        n_feedback_windows = math.ceil(input_shape[1] / self.feedback_window_size) if self.training else 1
        for n_feedback_tokens in range(0, n_feedback_windows, self.feedback_window_size):

            if self.training:
                # set input shape to feedback window size to format attention masks correctly
                seq_length = input_shape[1]
                seq_length = min(n_feedback_tokens + self.feedback_window_size, seq_length)
                input_shape = (input_shape[0], seq_length)

            ### CHANGE ABOVE

            batch_size, seq_length = input_shape

            # required mask seq length can be calculated via length of past
            mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

            if use_cache is True:
                assert self.is_decoder, ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(
                    self
                )

            if attention_mask is None:
                attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
            if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
                encoder_seq_length = encoder_hidden_states.shape[1]
                encoder_attention_mask = torch.ones(
                    batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
                )

            # initialize past_key_values with `None` if past does not exist
            if past_key_values is None:
                past_key_values = [None] * len(self.block)

            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

            if self.is_decoder and encoder_attention_mask is not None:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = None

            # Prepare head mask if needed
            head_mask = self.get_head_mask(head_mask, self.config.num_layers)
            encoder_head_mask = self.get_head_mask(encoder_head_mask, self.config.num_layers)
            present_key_value_states = () if use_cache else None
            all_hidden_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None
            all_cross_attentions = () if (output_attentions and self.is_decoder) else None
            position_bias = None
            encoder_decoder_position_bias = None

            for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
                layer_head_mask = head_mask[i]
                encoder_layer_head_mask = encoder_head_mask[i]
                # Model parallel
                if self.model_parallel:
                    torch.cuda.set_device(hidden_states.device)
                    # Ensure that attention_mask is always on the same device as hidden_states
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(hidden_states.device)
                    if position_bias is not None:
                        position_bias = position_bias.to(hidden_states.device)
                    if encoder_hidden_states is not None:
                        encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                    if encoder_extended_attention_mask is not None:
                        encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                    if encoder_decoder_position_bias is not None:
                        encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                    if layer_head_mask is not None:
                        layer_head_mask = layer_head_mask.to(hidden_states.device)
                    if encoder_layer_head_mask is not None:
                        encoder_layer_head_mask = encoder_layer_head_mask.to(hidden_states.device)
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                ### CHANGE BELOW

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    encoder_layer_head_mask=encoder_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    n_feedback_tokens=n_feedback_tokens
                )

                ### CHANGE ABOVE

                # layer_outputs is a tuple with:
                # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                hidden_states, present_key_value_state = layer_outputs[:2]

                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention weights),
                # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                position_bias = layer_outputs[2]
                if self.is_decoder and encoder_hidden_states is not None:
                    encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
                # append next layer key value states
                if use_cache:
                    present_key_value_states = present_key_value_states + (present_key_value_state,)

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[3],)
                    if self.is_decoder:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

                # Model Parallel: If it's the last layer for that device, put things on the next device
                if self.model_parallel:
                    for k, v in self.device_map.items():
                        if i == v[-1] and "cuda:" + str(k) != self.last_device:
                            hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

    self.forward = alt_forward_feedback

    return self
"""
