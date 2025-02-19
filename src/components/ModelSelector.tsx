'use client';

import { Fragment, memo, type ReactElement } from 'react';
import { Listbox, ListboxButton, ListboxOption, ListboxOptions, Transition } from '@headlessui/react';
import { CheckIcon, ChevronUpDownIcon } from '@heroicons/react/20/solid';
import { motion } from 'framer-motion';

export interface Model {
  readonly id: string;
  readonly name: string;
  readonly provider: 'tensorflow' | 'huggingface';
}

interface ModelSelectorProps {
  readonly models: readonly Model[];
  readonly selectedModel: Model;
  readonly onModelChange: (model: Model) => void;
}

const ModelOption = memo(({ 
  model, 
  selected 
}: { 
  model: Model; 
  selected: boolean;
}): ReactElement => (
  <motion.div
    initial={false}
    animate={{ scale: selected ? 1 : 0.98 }}
    className={`flex items-center relative cursor-default select-none py-3 pl-10 pr-4 ${
      selected ? 'bg-[#4B5320]/10 text-[#4B5320]' : 'text-gray-900'
    }`}
  >
    <span
      className={`block truncate text-base ${
        selected ? 'font-semibold' : 'font-normal'
      }`}
    >
      {model.name}
    </span>
    {selected && (
      <span
        className="absolute inset-y-0 left-0 flex items-center pl-3 text-[#4B5320]"
      >
        <CheckIcon className="h-5 w-5" aria-hidden="true" />
      </span>
    )}
    <span className="ml-auto text-sm text-gray-500">
    &nbsp;{model.provider === 'tensorflow' ? 'TensorFlow.js' : 'Hugging Face'}
    </span>
  </motion.div>
));

ModelOption.displayName = 'ModelOption';

function ModelSelector({
  models,
  selectedModel,
  onModelChange,
}: ModelSelectorProps): ReactElement {
  return (
    <div className="w-full min-w-[300px] relative z-[100]">
      <Listbox value={selectedModel} onChange={onModelChange}>
        {({ open }) => (
          <div className="relative mt-1">
            <ListboxButton className="relative w-full cursor-default rounded-lg bg-white py-3 pl-4 pr-10 text-left border-2 border-gray-200 hover:border-[#4B5320] focus:outline-none focus-visible:border-[#4B5320] focus-visible:ring-2 focus-visible:ring-[#4B5320]/20 transition-all duration-200 shadow-sm hover:shadow-md">
              <span className="block truncate text-base font-medium text-gray-900">
                {selectedModel.name}
              </span>
              <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-3">
                <ChevronUpDownIcon
                  className="h-6 w-6 text-[#4B5320]"
                  aria-hidden="true"
                />
              </span>
            </ListboxButton>
            <Transition
              as={Fragment}
              show={open}
              leave="transition ease-in duration-100"
              leaveFrom="opacity-100"
              leaveTo="opacity-0"
            >
              <ListboxOptions className="absolute z-[100] mt-2 max-h-28 w-full overflow-auto rounded-lg bg-white py-2 text-base shadow-lg ring-1 ring-black/5 focus:outline-none">
                {models.map((model) => (
                  <ListboxOption
                    key={model.id}
                    value={model}
                    className="relative cursor-pointer transition-colors duration-150"
                  >
                    {({ selected }) => (
                      <ModelOption
                        model={model}
                        selected={selected}
                      />
                    )}
                  </ListboxOption>
                ))}
              </ListboxOptions>
            </Transition>
          </div>
        )}
      </Listbox>
    </div>
  );
}

export default memo(ModelSelector); 