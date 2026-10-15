interface SwitchProps {
  checked: boolean;
  onChange: (on: boolean) => void;
  label: string;
  disabled?: boolean;
  title?: string;
}

export default function Switch({ checked, onChange, label, disabled, title }: SwitchProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      className="ask-switch"
      disabled={disabled}
      title={title}
      onClick={() => onChange(!checked)}
    >
      <span className="ask-switch__track" aria-hidden="true">
        <span className="ask-switch__thumb" />
      </span>
      <span className="ask-switch__label">{label}</span>
    </button>
  );
}
